# Route-strength calibration (Split MNIST, test)

Task chosen by route mass (raw), by route mass divided by its own-task value recorded at the end of that task (calibrated, legal), or recomputed after training (calib_now, not legal). Then classified with that task's route. Mean ± SD over 3 seed means.

| layers | gate loss | image only (soft) | raw: all-10 | raw: task right | calibrated: all-10 | calibrated: task right | calib_now: all-10 | last task picked (calibrated) | true route |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | off | 69.6 ± 0.4 | 57.4 ± 0.5 | 57.7 ± 0.5 | 51.6 ± 0.7 | 52.0 ± 0.9 | 64.4 ± 0.9 | 42.2 ± 3.5 | 98.5 ± 0.5 |
| 1 | 0.1 full | 28.1 ± 1.7 | 27.9 ± 1.2 | 28.2 ± 1.3 | 33.3 ± 3.3 | 33.7 ± 3.4 | 64.6 ± 2.2 | 79.2 ± 8.7 | 97.9 ± 0.3 |
| 1 | 0.1 up | 55.6 ± 3.0 | 35.8 ± 5.4 | 36.2 ± 5.5 | 38.3 ± 3.2 | 39.0 ± 3.4 | 65.8 ± 3.4 | 68.6 ± 8.1 | 98.1 ± 0.8 |
| 1 | 1 full | 20.7 ± 0.1 | 20.6 ± 0.2 | 20.9 ± 0.2 | 21.4 ± 0.6 | 21.7 ± 0.6 | 62.2 ± 3.4 | 98.4 ± 0.9 | 97.5 ± 0.2 |
| 1 | 1 up | 19.1 ± 1.3 | 20.7 ± 0.2 | 21.0 ± 0.2 | 22.1 ± 0.7 | 22.4 ± 0.7 | 49.1 ± 1.7 | 97.3 ± 0.8 | 93.8 ± 3.5 |
| 2 | off | 50.0 ± 4.5 | 48.1 ± 4.8 | 48.5 ± 4.6 | 44.7 ± 0.1 | 45.3 ± 0.4 | 55.1 ± 2.8 | 49.6 ± 3.0 | 98.0 ± 0.4 |
| 3 | off | 40.3 ± 3.9 | 47.5 ± 2.5 | 48.1 ± 3.0 | 40.5 ± 2.8 | 41.4 ± 3.3 | 52.8 ± 2.7 | 46.2 ± 18.8 | 97.5 ± 0.7 |
