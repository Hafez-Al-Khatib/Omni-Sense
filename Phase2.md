# Phase 2: The Data Engineering Reality & Physics Alignment

**Objective**: Map the theoretical YAMNet acoustic pipeline to the physical reality of the dataset and edge hardware.

## 1. Discovering the Sensor Physics (CSV vs. WAV)

We discovered that our dataset consists of raw time-series CSV files, not standard audio files.

- The Realization: We are not analyzing airborne sound via microphones. We are analyzing structure-borne vibrations captured by piezoelectric accelerometers.

- The Engineering Solution: We designed an automated Python pipeline using pandas and scipy.io to normalize the raw voltage data and encode it as 16-bit PCM .wav files. To a Convolutional Neural Network (YAMNet), a 1D time-series vibration array is mathematically indistinguishable from audio.

## 2. Validating the Bandwidth Tradeoff

- The Discovery: An analysis of the CSV timestamp deltas (3.91E-05 seconds) proved the raw physical sensors sampled at 25,600 Hz (25.6 kHz).

- The Implementation: This perfectly validated our theoretical bandwidth tradeoff. We integrated librosa.resample into our pipeline to downsample the 25.6kHz data to YAMNet's required 16kHz, reducing data payload size before it hits the External Endpoint (EEP).

## 3. Rejecting "Airborne" Augmentation

- The Decision: We initially considered overlaying open-source urban noise (ESC-50 dataset) onto our leaks.

- The Pivot: We realized this violates the physics of our sensors. Accelerometers reject airborne noise. Mixing microphone data with accelerometer data would corrupt the model. We chose the "Purist Route"—training strictly on clean structural data and relying entirely on our Isolation Forest to halt inference when anomalous ground-borne vibrations occur.

4. Dynamic Metadata Extraction

- The Solution: Because the dataset folders are named after the physical topologies (e.g., Branched/Circumferential Crack), we designed a pathlib script to automatically extract these labels and generate the JSON manifest required for the XGBoost (IEP 2) classifier.