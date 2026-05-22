# Data Preparation

SafeSABR uses Starlink throughput measurements derived from StarNet:

https://github.com/ConnectedSystemsLab/StarNet

The measurement data are not included in this repository. To reproduce the Starlink ABR experiments, prepare processed throughput files with at least two columns:

- `timestamp`: measurement timestamp.
- `throughput`: downlink throughput in Mbps.

Place them under:

```text
data/starnet_pkl/
├── us/dataset_tp_sat.pkl
├── osn/dataset_tp_sat.pkl
└── vic/dataset_tp_sat.pkl
```

Then convert them into SABR replay traces:

```bash
python tools/prepare_starlink_traces.py --source-root data/starnet_pkl
```

The converter writes traces to:

```text
safesabr/video_trace/trace/starlink/
├── us/{train,calib,test}
├── osn/{train,calib,test}
└── vic/{train,calib,test}
```

Each generated trace is a two-column text file:

```text
relative_time_seconds throughput_mbps
```

The default split is 70% training, 15% calibration, and 15% test within each location.

