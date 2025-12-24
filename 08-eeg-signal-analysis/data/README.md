# Data Directory

## Dataset: EEG Motor Movement/Imagery Dataset

**Source:** PhysioNet / MNE-Python built-in dataset
**Citation:** Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004)

## Structure

```
data/
├── raw/                    # Downloaded EEG files (.edf)
│   └── MNE-eegbci-data/    # Auto-created by MNE
├── processed/              # Preprocessed epochs
│   ├── epochs_motor.fif    # Motor imagery epochs
│   └── epochs_rest.fif     # Rest vs movement epochs
└── sample/                 # Small demo subset
    └── sample_epochs.fif   # 50 epochs for quick testing
```

## Download Instructions

Run the download script:
```bash
python scripts/download_data.py
```

This will:
1. Download EEG data for 10 subjects from PhysioNet
2. Create sample subset for demos
3. Generate metadata file

## Dataset Details

| Property | Value |
|----------|-------|
| Subjects | 10 (can extend to 109) |
| Channels | 64 EEG electrodes |
| Sampling Rate | 160 Hz |
| Tasks | Motor imagery (left/right hand), Rest |
| Trials per subject | ~45 per task |

## Files Downloaded

For each subject (S001-S010):
- `S00XR04.edf` - Left/right fist imagery
- `S00XR08.edf` - Left/right fist imagery
- `S00XR12.edf` - Left/right fist imagery
- `S00XR03.edf` - Open/close fists (real movement)
- `S00XR07.edf` - Open/close fists (real movement)
- `S00XR11.edf` - Open/close fists (real movement)

## Event Codes

| Code | Description |
|------|-------------|
| T0 | Rest |
| T1 | Left fist (imagery or movement) |
| T2 | Right fist (imagery or movement) |

## License

PhysioNet Open Data License
