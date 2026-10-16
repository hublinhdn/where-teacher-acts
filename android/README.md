# On-device latency (Android)

Times the deployed graphs of the deployment-cost benchmark on a USB-connected Android phone, with the same
ONNX Runtime version (1.26.0, CPU execution provider) and the same protocol as the desktop benchmark
(`pipeline/27_bench_latency.py`): per graph and thread setting (1 and 4 intra-op threads), a new session, 30
warm-up and 200 timed runs of one 224 x 224 image; five rounds round robin over all graphs; the median over
rounds of the per-round medians. The program runs on the phone's four fastest cores, and the battery
temperature and thermal status are logged around every round.

## One-time setup (laptop)

1. Android SDK platform-tools (`adb`) and an NDK (r27 or later), for example from Android Studio's SDK manager.
2. The ONNX Runtime Android package, unzipped (it is not in git):

   ```
   mkdir -p android/vendor/ort-1.26.0-aar android/vendor/ort-1.26.0
   curl -fL -o android/vendor/ort-1.26.0-aar/onnxruntime-android-1.26.0.aar \
     https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.26.0/onnxruntime-android-1.26.0.aar
   unzip -q android/vendor/ort-1.26.0-aar/onnxruntime-android-1.26.0.aar -d android/vendor/ort-1.26.0
   ```

   SHA-1 of the package: `8064b1e65f35a3cad2214324c2dbeceadcc6ecc7` (as published next to it on Maven Central).
3. Build: `bash android/build.sh <ndk_dir> android/vendor/ort-1.26.0 android/build`

## The graphs

On the machine that holds the checkpoints and the calibration images:
`python pipeline/38_export_latency_graphs.py` writes the 28 graphs (7 students x A and QAKD(B) x FP32 and
INT8) and a manifest with sizes and SHA-256 to `reports_qj/bench/main/latency_graphs/`. Copy that folder to
the laptop.

## The phone

Enable developer options and USB debugging, connect it, and accept the debugging prompt (`adb devices` lists
it as `device`). Before measuring: battery above 80 % and kept on the charger, airplane mode, battery saver
off, no other apps, phone at room temperature.

## Run

```
python android/run_android_bench.py --graphs reports_qj/bench/main/latency_graphs
```

Writes `reports_qj/bench/main/latency_android.csv` (the desktop file's columns plus the device),
`latency_android_rounds.csv` and `latency_android_device.json`; refuses to overwrite them unless `--force`.
A quick check with one round: add `--rounds 1 --out-dir <scratch folder>`. Everything is pushed to
`/data/local/tmp/qj_bench` on the phone and removed at the end.
