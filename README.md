# CLIFT: Contrastive LiDAR Instance Features for 3D Pedestrian Tracking

Accepted at **ACCV 2026**.

A LiDAR-only 3D pedestrian tracker. Instance embeddings are pooled from the intermediate
voxel features a detector already produces, trained with an instance-level contrastive
objective, and matched across frames by cosine similarity and the Hungarian algorithm —
no motion model, no camera fusion, no separate re-identification network.

On the JRDB 3D tracking leaderboard: **1st on HOTA (38.58) and IDF1 (37.99)**, 13.04 and
9.30 points above the next published method; 2nd on MOTA and OSPA.

**Project page:** https://hermak56.github.io/CLIFT/

## Code and data

Training and inference code, configuration files, pre-trained checkpoints and evaluation
scripts will be released here after the camera-ready version is finalised.
