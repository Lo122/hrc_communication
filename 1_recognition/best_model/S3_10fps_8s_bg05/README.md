# S3_10fps_8s_bg05

Deployable recognition model. Retrained on all 15 subjects for 4 epochs (epoch count chosen on held-out subjects [2, 7]).

Expected accuracy (15-fold leave-one-subject-out of the same settings): macro-F1 0.499 +/- 0.086.

## Files

| File | Contents |
|---|---|
| model_weights.pth | state_dict (`models.build('gru', 251, n_tasks=7)`) |
| standardization.npz | `mean`, `std`, `columns` -- one entry per input column |
| standardization_by_panel.npz | same stats keyed `<panel>_mean` / `<panel>_std` |
| feature_selection.json | panels, joints, exact column order, transforms, window |
| config.json | classes, heads, training settings, trigger, provenance |
| model_bundle.pt | all of the above in one file |

## Input

- 251 columns in the order of `feature_selection.json` -> `column_order`.
- polar_azimuth is converted to sin/cos BEFORE standardisation; ratios are clipped to [0, 4].
- Window: 80 samples, one every 3 source frames = 10 Hz. Feed the model at that rate.

## Output

- task: 7 independent sigmoid logits, classes ['Pull Cables', 'Lift', 'Place', 'Align', 'Screw', 'Connect Cables', 'Clamp Coupling'].
- bg: idle probability (sigmoid).

## Not yet done

- The trigger formula in config.json has not been re-validated on this model.
- 1_recognition/recognition_manager.py still loads the older 2-head AssistLSTM and needs updating for this 4-head model.
