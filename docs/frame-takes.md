# Recorded Frame takes

Handspotter trains on recordings from the Steam Frame's four tracking cameras.
Recording happens on the headset with tooling outside this repository. This
page specifies the take format the [labeler](../tools/labeler/README.md)
ingests.

A take is a directory with two files.

## `tracking.nut`

A NUT container with four video tracks, one per camera.

- Each track is `rawvideo` in 8-bit `gray` at the camera's native sensor
  resolution. Each packet holds exactly one frame.
- Each track's `title` metadata names its camera: `slam_left`, `slam_right`,
  `upper_left` or `upper_right`.
- Each track's time base is 1/1,000,000 s, and packet timestamps increase
  strictly within a track.

## `timestamps.csv`

One row per frame, with this header:

```
group,track,camera,pts_us,sensor_raw_s,end_exposure_raw_s,arrival_raw_s,exposure_s,observer_mono_ns,clock_raw_ns,clock_mono_ns,clock_uncertainty_ns
```

Rows are ordered by synchronized camera group. Every group has exactly one row
per camera, and each track's packets appear in the same order as that camera's
rows.

| Column | Meaning |
| --- | --- |
| `group` | Group sequence number. It never decreases. |
| `track` | Index of the camera's track in `tracking.nut` |
| `camera` | Camera name, matching the track title |
| `pts_us` | Packet timestamp in microseconds. It must match the packet to within 2 µs. |
| `sensor_raw_s` | Sensor timestamp in seconds, headset raw clock |
| `end_exposure_raw_s` | End of exposure in seconds, headset raw clock |
| `arrival_raw_s` | Frame arrival in seconds, headset raw clock |
| `exposure_s` | Exposure duration in seconds |
| `observer_mono_ns` | Monotonic time the group was observed, in nanoseconds |
| `clock_raw_ns`, `clock_mono_ns` | One paired reading of the raw and monotonic clocks, in nanoseconds |
| `clock_uncertainty_ns` | Uncertainty of that pairing, in nanoseconds |

## Calibration

Copy `/persist/xrservice.json` and `/persist/device_config.json` from the
headset that recorded the take and keep them with its takes. Dataset
preparation and the labeler's handedness tools use them. Native takes use
about 10 GB per minute.
[Labeling and dataset preparation](handspotter-training.md)
