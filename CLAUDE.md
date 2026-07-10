# poker44-xdev — UID 66 Miner

## Identity
- **Subnet**: Bittensor 126 (Poker44)
- **UID**: 66 | **Hotkey**: zero-1 | **Port**: 8094
- **Git account**: ultimate8888 (`ultimate90826@gmail.com`)
- **GitHub repo**: https://github.com/Ultimate8888/poker44-xdev
- **Token**: stored in `.secrets` (chmod 600, gitignored — NEVER commit)

## What this repo is
Independent second miner for UID 66, completely separate from `Poker44-subnet` (UID 123 / SerGem811). Uses a lighter architecture to avoid copycat/duplicate penalties.

- **Model**: Single `HistGradientBoostingClassifier` + `IsotonicRegression` calibration
- **Features**: 25 temporal/behavioral features (`xdev/features.py`)
- **Scoring**: `sigmoid_score(p, t_star=0.48, sharpness=10.0)`
- **Within-batch normalization**: yes

## Key files
| File | Purpose |
|---|---|
| `neurons/miner.py` | Main miner entry point (XdevMiner class) |
| `xdev/features.py` | 25-feature extractor (`extract_xdev_features`) |
| `xdev/model.py` | XdevModel wrapper + `sigmoid_score` |
| `models/xdev_v1.joblib` | Trained model artifact |
| `xdev/manifest.py` | Compliance manifest |
| `scripts/train.py` | Training script |

## Runtime dependency
Requires `Poker44-subnet` on the Python path for the base package:
```bash
PYTHONPATH=/root/work/Poker44-subnet python neurons/miner.py ...
```
The miner sets this automatically via `sys.path.insert`.

## PM2 process
```bash
pm2 start poker44_xdev       # start
pm2 stop poker44_xdev        # stop
pm2 restart poker44_xdev     # restart
pm2 logs poker44_xdev        # logs
```

## Push to GitHub
```bash
source .secrets
git push "https://Ultimate8888:${GITHUB_TOKEN}@github.com/Ultimate8888/poker44-xdev.git" main
```

## Security rules
- NEVER commit `.secrets` or any token
- NEVER push to `Poker44/Poker44-subnet` upstream
- NEVER touch UID 123 (Poker44-subnet / SerGem811) from this repo

## Synapse fields (DetectionSynapse)
```python
synapse.chunks      # input: List[List[dict]]  — read this
synapse.risk_scores # output: List[float]       — write this
synapse.predictions # output: List[bool]        — write this (score >= 0.5)
```
