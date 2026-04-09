# T-IMPACT Dataset

Link to dataset: https://drive.google.com/file/d/1yGBKB_xJoicJ-qMj__o2Ae5M9gp4nTzU/view?usp=sharing

We release all generation files for the timpact dataset in this repository.

The order to run these files is 

1. extract_anchors_qwen.py 

2. ground_anchors_dino.py

3. segment_from_boxes_sam.py

4. extract_editable_anchors.py

5. generate_edit_suggestions_qwen_v2.py

6. run_image_edits.py

7. timpact_final_builder.py

## Supplementary 

- Line 431, Section 3
  
In this first release, the raw formulation should be interpreted as a structured, human-informed prior over contextual impact rather than a final fully optimized estimator. The component weights encode an explicit judgment about what most strongly changes narrative meaning: higher-impact objects and stronger contextual incongruity are given greater influence because they more directly alter perceived intent, blame, legitimacy, or threat, while contradiction and salience provide supporting semantic and perceptual evidence, and visibility is intentionally down-weighted because conspicuity alone does not guarantee interpretive harm. Importantly, changing these weights does not simply rescale the final score; it can also change the relative ordering of examples, shift the calibration map, alter where low/medium/high cutoffs fall, and therefore affect how manipulations are perceived and distributed in the released benchmark. In turn, these changes influence downstream edit generation and evaluation, since the score governs which edits are treated as mild reframing versus stronger semantic distortion, and thus affects severity targeting, bin balance, and model-learning signals. To better align this structured prior with perceived severity, we apply isotonic regression on a limited overlapping subset of human ratings, preserving ordinal structure while avoiding a stronger parametric assumption. We therefore treat the calibrated continuous score as the primary supervision target, while the released low/medium/high labels serve as operational buckets for benchmarking rather than as a claim that severity is naturally discrete or intrinsically class-balanced. Future releases will revisit both the weighting scheme and thresholding strategy using larger expert-rated calibration sets and broader cross-dataset validation.

