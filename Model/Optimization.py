import optuna
from Training_Amplitude_Phase import *
import csv,json,time

def prepare_data(train_path, val_path,training_strategy,use_cuda):
    train_dataset = makeDataset(train_path, use_cuda)
    val_dataset = makeDataset(val_path, use_cuda)
    if training_strategy==Training_Strategy.ONE_VISIBILITY:
            raw_dataset = train_dataset
    
            train_count = min(256, len(raw_dataset) - 1)
            val_count = min(64, len(raw_dataset) - train_count)
    
            if train_count < 1 or val_count < 1:
                raise RuntimeError("Not enough windows for testing")
    
            train_dataset = Subset(
                raw_dataset,
                list(range(train_count))
            )
            val_dataset = Subset(
                raw_dataset,
                list(range(train_count, train_count + val_count))
            )
    
    elif training_strategy == Training_Strategy.POOLED_STRATIFIED_SPLIT:
            train_parts = []
            val_parts = []
    
            for path in train_path:
                dataset = makeDataset([path], use_cuda)
    
                if len(dataset) < 40:
                    raise RuntimeError(
                        f"Not enough windows in {path}"
                    )
    
                train_parts.append(
                    Subset(dataset, range(0, 32))
                )
    
                val_parts.append(
                    Subset(dataset, range(32, 40))
                )
    
            train_dataset = ConcatDataset(train_parts)
            val_dataset = ConcatDataset(val_parts)
        
    elif training_strategy == Training_Strategy.ONE_VISIBILITY_DISTRIBUTED:
            dataset = train_dataset
            n = len(dataset)
    
            split = int(0.8 * n)
    
            if split < 32 or n - split < 8:
                raise RuntimeError("Not enough windows for distributed test")
    
            train_indices = torch.linspace(
                0, split - 1, steps=32
            ).round().long().tolist()
    
            val_indices = torch.linspace(
                split, n - 1, steps=8
            ).round().long().tolist()
    
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
    
            print("Total available windows:", n)
            print("Training indices:", train_indices)
            print("Validation indices:", val_indices)
        
    elif training_strategy == Training_Strategy.ONE_VISIBILITY_FULL:
            raw_dataset = train_dataset
    
            total_windows = len(raw_dataset)
            split = int(0.8 * total_windows)
    
            if split == 0 or split == total_windows:
                raise RuntimeError("Not enough windows for splitting")
    
            train_dataset = Subset(
                raw_dataset,
                range(0, split)
            )
    
            val_dataset = Subset(
                raw_dataset,
                range(split, total_windows)
            )
    
            print("Total windows:", total_windows)
            print("Training windows:", len(train_dataset))
            print("Validation windows:", len(val_dataset))
    
    
            
    elif training_strategy not in (Training_Strategy.FIVE_FOLD_CROSS,
                                       Training_Strategy.LEAVE_ONE_OUT):
            raise ValueError("Unknown Strategy or didn't select strategy")
    
    
    standardizer = VisibilityStandardizer()
    
    
    num_stats_samples = min(32, len(train_dataset))
    
    indices = torch.linspace(
                0,
                len(train_dataset) - 1,
                steps=num_stats_samples
            ).long()
    
    stats_samples = [train_dataset[i] for i in indices.tolist()]
    
    statistics_data = torch.cat([s["vis"] for s in stats_samples], dim=0)
    standardizer.fit(statistics_data)
    del statistics_data
    
    uvw_scale = None
    if USE_UVW:
            uvw_scale = (
                torch.cat([s["uvw"] for s in stats_samples], dim=0)
                .float().abs().amax(dim=(0, 1))   
                .clamp_min(1e-6)
            )
            print("UVW scale:", uvw_scale)
    del stats_samples
    
    val_standardizer = standardizer
    if training_strategy == Training_Strategy.LEAVE_ONE_OUT:
            val_indices = torch.linspace(
                0, len(val_dataset) - 1, steps=min(32, len(val_dataset))
            ).long().tolist()
            val_standardizer = VisibilityStandardizer()
            val_standardizer.fit(
                torch.cat([val_dataset[i]["vis"] for i in val_indices], dim=0)
            )
            print("Validation std / training std (real):",
                  (val_standardizer.real_std / standardizer.real_std).tolist())
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("One of the datasets is empty. Check file paths and time-window settings.")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "standardizer": standardizer,
        "val_standardizer": val_standardizer,
        "uvw_scale": uvw_scale,
    }
def run_training(config, data, device, output_dir=None, trial=None):

    torch.manual_seed(SEED)
    history_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        history_path = output_dir / "history.csv"
        if not history_path.exists():
            with open(history_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_nmse", "val_snr_db",
                                        "nmse_xx", "nmse_xy", "nmse_yx", "nmse_yy", "seconds"])

                
    transform = PolarizationStandardization()

    train_dataset = TransformDataset(data["train"], transform, data["standardizer"], data["uvw_scale"],
                                frequencies=FREQS_HZ, max_shift=config["max_shift"])
    
    val_dataset = TransformDataset(data["val"], transform, data["val_standardizer"], data["uvw_scale"],
                              frequencies=FREQS_HZ, max_shift=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = makeModel(device, Model_Type.AUTOENCODER_BASE,
                      n_channels=config["n_channels"], kernel_size=config["kernel_size"])
    criterion = PhysicalComplexMSE(data["standardizer"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])

    best_nmse = float("inf")
    for epoch in range(config["epochs"]):

        start = time.time()
        model.train()
        
        total_train_loss = 0.0
        total_train_batches = 0
        
        for batch in train_loader:
                    
            x = to_baselines(batch.to(device))   
        
            if not torch.isfinite(x).all():
                    raise RuntimeError(f"Non-finite input at epoch {epoch + 1}")
        
            optimizer.zero_grad()
        
            outputs = model(x)                     
            target = x[:, :N_VIS_CHANNELS]
            if not torch.isfinite(outputs).all():
                    raise RuntimeError(
                            f"Non-finite model output at epoch {epoch + 1}"
                        )
                    
        
            loss = criterion(outputs, target)
            if not torch.isfinite(loss):
                    raise RuntimeError(
                            f"Non-finite loss at epoch {epoch + 1}: {loss.item()}"
                        )
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                                raise RuntimeError(
                                    f"Non-finite gradient in {name} "
                                    f"at epoch {epoch + 1}"
                                )
            torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                        error_if_nonfinite=True,
                    )
                    
            optimizer.step()
        
            total_train_loss += loss.item()
            total_train_batches += 1
        
        avg_train_loss = total_train_loss / max(total_train_batches, 1)
        val_loss, val_nmse, val_snr, val_nmse_pol = validate_model(
            model, val_loader, criterion, device, data["val_standardizer"])
        
        seconds = time.time() - start

        pol = val_nmse_pol.tolist()
        print(f"Epoch {epoch + 1}/{config['epochs']} | train loss {avg_train_loss:.6f} | "
              f"val NMSE {val_nmse:.6f} | per pol [XX XY YX YY] "
              f"{' '.join(f'{v:.4f}' for v in pol)} | {seconds / 60:.1f} min", flush=True)

        if history_path is not None:
            with open(history_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, avg_train_loss, val_loss, val_nmse, val_snr,
                                        *pol, round(seconds, 1)])

        if trial is not None:                   
            trial.report(val_nmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_nmse < best_nmse:
            best_nmse = val_nmse
            if output_dir is not None:          
                torch.save({"model_state_dict": model.state_dict(), "config": config},
                           output_dir / "fold_1_best.pt")
    return best_nmse

if __name__ == "__main__":
    device = torch.device("cpu")

    paths = sorted({
        str(Path(p).resolve())
        for arg in sys.argv[1:]
        for p in (glob.glob(arg) or [arg])
        if Path(p).is_file()
    })
    tune_obs = os.environ.get("VAL_OBS", "s41")           # tune on s41, keep s40 for the final report
    val_paths = [p for p in paths if f"_{tune_obs}_" in Path(p).name]
    train_paths = [p for p in paths if p not in val_paths]
    assert len(val_paths) == 1 and len(train_paths) >= 1, (val_paths, train_paths)
    print(f"Tuning: validate on {tune_obs}, train on {len(train_paths)} observation(s)")

    data = prepare_data(train_paths, val_paths, Training_Strategy.LEAVE_ONE_OUT, use_cuda=False)

    results_root = Path(os.environ.get(
        "OPTUNA_DIR", str(PROJECT_ROOT / "results" / "optuna_results")))
    results_root.mkdir(parents=True, exist_ok=True)
    trial_epochs = int(os.environ.get("TRIAL_EPOCHS", "6"))

    def objective(trial):
        config = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "max_shift":    trial.suggest_float("max_shift", 0.0, 0.1),
            "n_channels": trial.suggest_categorical("n_channels", [16, 32, 64]),
            "kernel_size":  trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "epochs":       trial_epochs,
        }
        trial_dir = results_root / f"trial_{trial.number:03d}"
        return run_training(config, data, device, output_dir=trial_dir, trial=trial)

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(results_root / "optuna_perbaseline_8x.log")))
    study = optuna.create_study(
        study_name="perbaseline_8x", storage=storage, direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(objective, timeout=float(os.environ.get("TIMEOUT_H", "6")) * 3600)
    print("Best:", study.best_value, study.best_params)
    