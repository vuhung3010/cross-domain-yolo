---
title: "Adaptive Gradient Reversal for Domain-Adaptive YOLO-G Object Detection Under Foggy Weather"
author: "Vu Hung"
date: "2026"
bibliography: references.bib
csl: ieee.csl
link-citations: true
geometry: margin=1in
fontsize: 11pt
---

# Abstract

Object detectors for autonomous driving often assume that training and deployment images follow similar visual distributions, but clear-weather detectors can degrade sharply in fog where contrast, illumination, and object boundaries change. We study unsupervised domain-adaptive YOLO-G detection from labeled Cityscapes to unlabeled Foggy Cityscapes, keeping the inference-time detector unchanged while modifying the training-time adaptation signal. The proposed analysis centers on adaptive gradient reversal (AdvGRL), which computes adversarial pressure from a detached domain-classifier loss, and evaluates where that pressure should enter the detector feature hierarchy. In our ablations, source-only YOLO-G obtains 0.37841 mAP@0.5, fixed image-level YOLO-G domain adaptation improves to 0.41195, and isolated deepest-feature AdvGRL reaches 0.39839. The best-performing configuration applies AdvGRL to multi-scale neck features, achieving 0.46665 mAP@0.5 and 0.28955 mAP@0.5:0.95, a gain of 0.08824 over source-only YOLO-G and 0.05470 over fixed image-level adaptation. RainMix-coupled triplet learning performs worse than the baselines both alone and when combined with AdvGRL, indicating that the synthetic rainy auxiliary domain is not complementary for this fog-adaptation setting. These results support a scoped conclusion: adaptive adversarial weighting is most effective when paired with feature levels that directly support multi-scale detection, while auxiliary-domain metric learning requires stronger validation before being treated as beneficial.

# 1. Introduction

## 1.1 Problem and Motivation

In safety-critical driving perception, detection models must generalize beyond the conditions represented in labeled training data. A detector trained on clear-weather Cityscapes can suffer a measurable performance drop when evaluated on Foggy Cityscapes, despite unchanged object categories and similar street-scene layouts [@cordts2016cityscapes; @sakaridis2018semanticfoggy]. Manual annotation of target-domain detection data is expensive, motivating unsupervised domain adaptation with labeled source images and unlabeled target images.

Object detection makes this adaptation problem harder than ordinary image classification. The model must align weather-dependent appearance while preserving spatially localized objectness, class discrimination, and bounding-box regression. Overly strong global alignment can remove useful detection cues, especially for small or low-contrast objects in fog. This tension motivates an adaptive adversarial method rather than a single fixed reversal strength.

The detector family studied here is YOLO-G, derived from YOLOv5-L [@wei2023yolog; @jocher2022yolov5]. The project does not propose a new inference-time detector architecture. Instead, it attaches training-time domain-adaptation objectives to YOLO-G representations and evaluates how adversarial weighting, feature placement, and auxiliary metric learning affect target-domain detection.

## 1.2 Objective

We ask whether adaptive adversarial weighting improves YOLO-G adaptation on Cityscapes -> Foggy Cityscapes and whether its effect depends on the feature level that receives the adversarial signal. This objective follows the structure of domain-adaptive detection papers: first establish the source-only and fixed-adaptation baselines, then test whether the proposed adaptation mechanism improves target-domain detection under controlled ablations.

## 1.3 Contributions

1. An AdvGRL-centered analysis showing that adaptive adversarial weighting is most effective in this study when applied to multi-scale neck features.
2. A controlled ablation design isolating source-only YOLO-G training, fixed image-level YOLO-G domain adaptation, adaptive weighting, feature-level placement, foreground gating, and auxiliary triplet learning.
3. A full evidence table reporting target-domain detection metrics and adaptation diagnostics for the completed Cityscapes -> Foggy Cityscapes runs.
4. A failure analysis showing that the RainMix-coupled triplet extension does not improve the AdvGRL-centered result and should be treated as an investigated limitation rather than the headline method.

## 1.4 Positioning

We build on YOLO-G, a YOLOv5-L-derived detector for cross-domain object detection, and treat the original image-level YOLO-G adaptation setting as the fixed-weight adversarial baseline. The proposed contribution is not a new detector backbone. It is a training-time adaptation study that evaluates adaptive gradient reversal, multi-scale feature placement, foreground gating, and RainMix-coupled triplet learning under one controlled Cityscapes -> Foggy Cityscapes protocol.

# 2. Related Work

## 2.1 YOLO-Style Object Detection

YOLO-style detectors remain a practical base for autonomous-driving perception because they combine a single-stage prediction pipeline with multi-scale detection heads that cover small, medium, and large objects in the same forward pass [@jocher2022yolov5]. In the YOLO-G lineage used here, the detector inherits the familiar backbone-neck-head structure from the YOLO family and adapts it to the cross-domain detection setting rather than redesigning detection from scratch [@wei2023yolog]. That choice matters because the methodological question is not whether a new detector backbone can outperform the old one, but whether a training-time adaptation strategy can improve the same detector under foggy-domain shift.

## 2.2 Unsupervised Domain Adaptation for Object Detection

Unsupervised domain adaptation for detection is harder than for classification because the model must preserve both category discrimination and localization quality while transferring from labeled source images to unlabeled target images. Feature alignment can help, but if it is applied too aggressively it may suppress cues needed for objectness or box regression. This tension is central to the present work: the detector must remain accurate on source semantics while becoming robust to the appearance shift introduced by fog.

A detector also contains several feature levels with different roles. Shallow or neck-level features retain spatial detail that can matter for small objects, while deeper representations carry stronger semantic abstraction. Domain alignment at only one level may therefore miss part of the detection problem. If fog changes local contrast, object boundary clarity, and background texture, then adapting only the deepest feature may not correct the feature distributions used directly by multi-scale prediction heads. This observation motivates evaluating both image-level adaptation and multi-scale neck adaptation rather than assuming one feature level is sufficient.

The detection setting also makes negative evidence important. A domain-adaptation objective can improve a global domain-confusion criterion while harming detection metrics. For example, a representation that is more domain-invariant may also be less discriminative for small or low-contrast objects. We therefore treat mAP, precision, recall, and training diagnostics as joint evidence. A method is not considered successful merely because it adds an adaptation loss; it must improve target-domain detection while remaining stable across the full training schedule.

Earlier domain-adaptive detection work also shows why feature placement and alignment granularity matter. DA-Faster R-CNN introduced image-level and instance-level adaptation for detection, demonstrating that aligning only global image features is insufficient when object instances carry their own domain shift [@chen2018dafaster]. Strong-Weak Distribution Alignment further argued that detection benefits from strong local alignment but weaker global alignment because entire images may contain different layouts and object mixtures across domains [@saito2019swda]. These two ideas motivate the present report's emphasis on where the adversarial signal enters YOLO-G: aligning detection-near multi-scale features is conceptually closer to local/instance-aware adaptation than relying only on one deepest image-level representation.

## 2.3 DANN and Gradient Reversal

Domain-adversarial neural networks introduced gradient reversal as a simple way to turn domain confusion into a training signal [@ganin2016dann]. A domain classifier is trained to separate source and target features, while the feature extractor receives the opposite gradient and learns representations that make the domains harder to distinguish. The original YOLO-G image-level domain-adaptation baseline follows that template. Its value is mainly comparative: it establishes whether ordinary fixed-weight adversarial alignment helps on Foggy Cityscapes before introducing adaptive weighting.

## 2.4 Adaptive Adversarial Weighting

Adaptive adversarial weighting extends the DANN idea by changing the reversal strength during training rather than keeping it constant. The AdvGRL design studied here follows the domain-adaptive foggy-weather detection work of Li et al. [@li2022domainadaptivefoggy]. AdvGRL is central to this study because it uses the current classifier state to determine how strongly the detector should be pushed toward domain confusion. That makes the adaptation pressure responsive to training progress. If the domain classifier is too uncertain or too strong, a fixed weight may either undertrain or destabilize the detector; an adaptive rule is intended to mitigate that mismatch.

The broader DA-Detect line of work later extended this idea to adverse-weather driving with joint image-level and object-level adaptation, difficult-instance adversarial mining, and auxiliary-domain metric regularization [@li2024dadetect]. That context matters for the scope of this report. The present study does not reproduce the full DA-Detect framework; it ports a narrower subset of adaptive-gradient and RainMix-coupled ideas into the YOLO-G training pipeline. Therefore, the negative triplet result reported here should be read as evidence about this YOLO-G formulation, not as a general rejection of every object-level or auxiliary-domain adaptation design.

## 2.5 Weather Robustness and Synthetic Domains

Foggy Cityscapes is a standard stress test for weather robustness because the target domain changes visibility, contrast, and background structure without changing the underlying driving scene semantics [@sakaridis2018semanticfoggy]. Synthetic rain or fog can be useful as a source of additional variation, but it is not guaranteed to approximate the target shift. This matters for RainMix: a synthetic rainy domain can plausibly serve as a contrastive auxiliary domain, yet it can also introduce statistics that differ too much from fog and harm adaptation instead of helping it.

# 3. Data

## 3.1 Source Domain: Cityscapes

Cityscapes provides the labeled source domain [@cordts2016cityscapes]. It contains urban street scenes with dense annotations for common driving objects and a clear-weather appearance that is useful for supervised detector pretraining. The precise split and image definitions are recorded in the supplementary package so that the paper can remain method-focused while still being reproducible.

## 3.2 Target Domain: Foggy Cityscapes

Foggy Cityscapes is the unlabeled target domain during adaptation and the evaluation target for the final metrics [@sakaridis2018semanticfoggy]. It preserves the Cityscapes semantic structure while adding fog-induced appearance shift, which makes it well suited for studying whether domain adaptation improves detection under reduced visibility. Target annotations are not used to train the adaptation losses; they are reserved for validation and reported metrics.

## 3.3 Auxiliary Domain: RainMix

RainMix is an auxiliary synthetic-weather domain used only when the triplet objective is active. It does not define a separate main-paper method row because its role is to provide the negative auxiliary distribution in the RainMix-coupled triplet formulation. The value of this auxiliary domain is therefore conditional: it matters only insofar as it participates in the triplet loss, and its utility must be judged by the ablation evidence rather than assumed from its presence.

The data-domain visualization shows representative source, target, and auxiliary-domain images used in the study. The visual contrast clarifies the adaptation problem: Cityscapes provides clear-weather supervision, Foggy Cityscapes reduces contrast and visibility while preserving the driving-scene layout, and RainMix introduces a different synthetic weather pattern. This is why the auxiliary domain is treated cautiously. It is visually related to adverse-weather driving, but it is not the same distribution as fog.

![Representative data domains used in the ablation study. Cityscapes supplies labeled clear-weather source images, Foggy Cityscapes supplies the unlabeled target domain during adaptation, and RainMix supplies the auxiliary synthetic-weather samples used only for the triplet extension.](figures/fig_data_domains.pdf){#fig:data-domains width=95%}

## 3.4 Preprocessing and Training Inputs

All compared runs use the same image size and general detector preprocessing so that method differences are not confounded by unrelated data handling. Source batches supply the supervised detection loss, target batches supply unlabeled adaptation signals, and auxiliary RainMix batches are concatenated only for the triplet variants. That loader structure matters because it preserves a clear separation between source supervision, target adaptation, and auxiliary negative-domain pressure.

The same loader discipline also makes the experimental interpretation cleaner. If a method improves target performance, the improvement cannot be explained by a different image resolution, a different training split, or a different data stream arrangement. Likewise, if a method fails, the failure is more likely attributable to the adaptation objective itself. This is particularly relevant for RainMix-coupled triplet learning, where the auxiliary domain is not merely extra data but a required component of the loss definition.

## 3.5 Split Design and Evaluation Logic

The source, target, and auxiliary roles are deliberately asymmetric. Cityscapes contributes labels and serves as the supervised reference distribution. Foggy Cityscapes contributes unlabeled target images during adaptation and labeled images only for validation and reported metrics. RainMix contributes unlabeled auxiliary samples only when the triplet objective is enabled. That asymmetry reflects the actual optimization structure and keeps the report's methodological claims aligned with the loss definitions.

This split design also supports fair comparison across methods. The source-only YOLO-G baseline does not see target or auxiliary data. The original YOLO-G image-level domain-adaptation baseline sees source and target but no RainMix. AdvGRL inherits the same source-target pairing while modifying the way adversarial pressure is computed. The full original method adds RainMix-coupled triplet learning on top of that. Because each row changes only one conceptual component at a time, the resulting ablation table can be interpreted as a sequence of controlled perturbations rather than as an uncontrolled mixture of settings.

## 3.6 Dataset Limitations

This study is limited to one principal source-target pair, which keeps the ablation clear but prevents broad claims across weather types or datasets. The auxiliary RainMix domain is synthetic and therefore imperfectly matched to fog; this mismatch is likely part of why the triplet extension underperforms. A stronger reproducibility package would also record full data manifests, counts, and generation scripts for every split so that the exact train/validation composition is transparent.

The limitation is methodological as well as empirical. The study evaluates whether one adaptation design helps under one target shift, not whether the design generalizes to every driving-weather benchmark. That narrower scope permits direct evidence tracing and avoids overstating the negative result on RainMix: the result is specific to this fog-domain task and should not be read as a universal verdict on all auxiliary-domain or metric-learning strategies.

# 4. Methods

## 4.1 Overall Architecture

This project studies domain adaptation as a training-method change around a YOLO-G / YOLOv5-L detector rather than as a new detector backbone. The detector retains the inherited YOLO-style pipeline: images pass through a convolutional backbone and multi-scale feature neck, and the detection head predicts objectness, class probabilities, and bounding-box coordinates at multiple spatial resolutions. Domain-adaptation objectives are attached to intermediate representations during training while the detector architecture remains fixed across comparisons.

The method separates detection prediction from adaptation optimization. The detector produces predictions and intermediate feature tensors, whereas domain classification, gradient reversal, auxiliary-domain processing, and adaptation losses are applied externally during optimization. This separation ensures that every comparison uses the same detector capacity and differs only in its training objective. Consequently, performance changes can be attributed to individual adaptation components rather than architectural differences.

The overall objective contains supervised detection loss on labeled source-domain Cityscapes images and, when applicable, one or more adaptation losses using unlabeled target-domain Foggy Cityscapes images. Target images influence the representation through adversarial or metric-learning objectives, but target annotations are excluded from training and reserved for evaluation.

This design creates a useful separation between detector capacity and adaptation behavior. The detector is not enlarged for one row and reduced for another. Instead, the comparison asks how the same detector responds when different training signals are attached to its representations. That is why the report emphasizes optimization dynamics, feature level, and objective interaction. The central question is not whether YOLO-G can detect objects in fog at all, but whether a controlled adaptation signal can improve its target-domain behavior without damaging the detection task.


This naming is important for interpretation. The report treats YOLO-G as the base cross-domain detector family, not as an anonymous backbone. The first two ablation rows therefore establish the YOLO-G reference points: source-only YOLO-G measures the detector without target-domain alignment, and YOLO-G image-level domain adaptation measures the original fixed-weight adversarial adaptation setting. AdvGRL is introduced after those rows as a training-dynamics modification to the YOLO-G adaptation signal.

The architecture can be read as three interacting systems. The first is the supervised detector, which must maintain objectness, localization, and classification accuracy. The second is the adversarial domain-classification system, which attempts to remove domain-specific information from selected features. The third is the optional auxiliary-domain metric-learning system, which imposes a geometric relation among source, target, and RainMix representations. The ablation study tests these systems separately because their gradients may cooperate, conflict, or dominate each other depending on where and when they are applied.

## 4.2 Source-Only Baseline

The source-only YOLO-G baseline trains the detector on labeled Cityscapes images without target-domain adaptation. It provides the reference point for all comparisons: every adaptation variant uses the same initialization, detector configuration, image resolution, and training schedule, differing only in the additional adaptation objective.

This baseline anchors both quantitative ablations and qualitative detection comparisons. It measures the unadapted detector's degradation under the clear-to-foggy domain shift and establishes whether each proposed adaptation strategy produces a genuine target-domain benefit.

## 4.3 Fixed Image-Level Domain Adaptation

The original YOLO-G image-level domain-adaptation baseline adds a binary domain classifier to intermediate detector features. The classifier learns to distinguish source from target representations, while a gradient reversal layer sends sign-reversed gradients into the detector. The resulting minimax objective encourages the feature extractor to preserve detection-relevant information while reducing domain-specific information.

During each adaptation iteration, the source batch contributes supervised detection loss and source-domain labels for the domain classifier. The target batch contributes target-domain labels only to the domain-classification objective. A lightweight convolutional domain head produces dense spatial logits, allowing alignment across feature-map locations rather than through only one globally pooled vector. The gradient reversal operation is identity-valued during the forward pass and multiplies the backward feature gradient by a fixed negative scalar.

This fixed-weight baseline is important because it exposes the limitation of applying one adversarial pressure level throughout training. Earlier short-run evaluations showed that full adversarial pressure can severely degrade detection performance when the detector representation is still changing. That behavior motivates adaptive gradient reversal as the main training-dynamics intervention.

## 4.4 AdvGRL: Main Method

AdvGRL is the main method. Its purpose is to adapt the adversarial gradient strength to the current state of the domain classifier. A fixed GRL weight assumes the same adversarial pressure is appropriate at every iteration. AdvGRL instead measures the classifier state first, computes a per-iteration $\lambda_{\mathrm{adv}}$, and then applies that value to the GRL-attached DA loss.

The implementation uses two DA-classifier passes. The first pass is detached from the detector features and computes a scalar classifier loss $L_c$. This pass measures how the domain classifier behaves without updating the detector backbone through the measurement itself. The second pass applies gradient reversal to the live features and computes the actual DA loss used for backpropagation. Thus, $L_c$ controls the adversarial weight, while the second pass supplies the optimization signal.

The effective weight follows the AdvGRL rule:

$$
\lambda_{\mathrm{adv}} =
\begin{cases}
\lambda_0 \min\!\left(\beta, \frac{1}{L_c}\right), & L_c \le \alpha, \\
\lambda_0, & L_c > \alpha.
\end{cases}
$$

Here $\lambda_0$ is the base reversal weight, $\beta$ caps the multiplicative factor, and $\alpha$ is the threshold separating the ordinary regime from the adaptive hard regime. The default $\alpha$ follows the adaptive-gradient design from domain-adaptive foggy-weather detection [@li2022domainadaptivefoggy], computed as binary cross-entropy with logits on predictions `[0.7, 0.3]` against labels `[1, 0]`. The resulting negative gradient scale performs reversal during the second pass.

The central hypothesis is that adversarial pressure should respond to the domain-classifier state rather than remain fixed throughout training. The evaluation therefore prioritizes the comparison between adaptive reversal and the original YOLO-G image-level domain-adaptation baseline, then tests whether feature placement changes the value of adaptive weighting. RainMix, triplet loss, and objectness gating are treated as extensions; the central claim concerns adaptive adversarial control and its interaction with feature placement.

The two-pass structure is important because it avoids using the same computation both to measure the classifier state and to update the detector representation. In the first pass, the detector features are treated as fixed with respect to the adaptive-weight calculation. In the second pass, the selected weight is applied to the reversed gradient that reaches the detector. This makes the adaptive signal easier to interpret: the classifier loss is a control variable, while the adversarial loss remains the optimization objective.

AdvGRL also changes the failure modes that should be inspected. A fixed reversal weight can fail by being too weak to align domains or too strong to preserve detection features. An adaptive rule can fail differently: it may react to the classifier loss in a way that amplifies noisy measurements, saturates the cap too often, or changes the training signal at the wrong representation level. For that reason, the report includes diagnostic curves rather than only final mAP. The method should be judged by both final target accuracy and whether its internal control variables behave plausibly over training.

The final results show that adaptive weighting alone is not sufficient. The deepest-feature AdvGRL row does not beat the original YOLO-G image-level domain-adaptation baseline, while multi-scale neck AdvGRL produces the strongest result. This does not invalidate the AdvGRL hypothesis; it refines it. Adaptive adversarial pressure appears most useful when applied to the feature levels that directly support detection across object scales. The method contribution is therefore best stated as adaptive multi-scale adversarial alignment rather than as a claim that any use of adaptive weighting is automatically superior.

## 4.5 Multi-Scale Neck Adaptation

The strongest AdvGRL-family setting applies adversarial alignment to the detector's multi-scale neck representations rather than only to one deepest feature. The neck contains feature levels used directly by the detection head for objects at different spatial scales. Aligning these representations can therefore affect the features that support small, medium, and large object predictions under fog.

This design is motivated by the structure of weather shift. Fog changes local contrast, boundary sharpness, and background texture, so the domain gap may remain visible at spatial feature levels that directly feed detection heads. A single deepest representation can be too abstract or too far removed from the multi-scale prediction path. Multi-scale neck adaptation tests whether adaptive adversarial control becomes more useful when it is applied closer to the detector features that determine target-domain predictions.

The multi-scale loss averages adaptation signals across the P3, P4, and P5 neck levels. This keeps the detector architecture unchanged for inference while broadening the training-time alignment signal. In the ablations, this feature-placement change is critical: isolated deepest-feature AdvGRL does not exceed the fixed image-level baseline, while multi-scale neck AdvGRL gives the strongest evaluated target-domain result.

## 4.6 Foreground/Objectness-Gated Extension

The foreground/objectness-gated extension applies domain adaptation more selectively to spatial regions likely to contain objects. Instead of aligning every spatial location equally, it uses the detector's objectness predictions to construct a detached foreground gate. This is a foreground-aware variant of multi-scale neck adaptation intended to reduce background-dominated alignment.

For each neck scale, objectness scores weight the spatial domain-classification loss, and the weighted loss is normalized by the total gate mass. Source and target features are assigned binary domain labels, and the final adaptation loss averages the three neck-scale losses. When combined with AdvGRL, the detached $L_c$ probe uses the same gated objective as the GRL-attached pass, so the adaptive weight is measured from the same type of loss that is later optimized.

The motivation for gating is strongest in dense driving scenes, where large portions of the image are road, sky, building facade, or texture that may not correspond to annotated objects. If the domain classifier mainly learns from those regions, the detector may spend adaptation capacity aligning background statistics rather than improving object detection. A foreground-weighted objective attempts to shift more of the adaptation pressure toward object-like spatial regions, but it also depends on the reliability of the detector's own objectness predictions.

The completed result reflects this trade-off. The foreground-gated extension performs better than the source-only YOLO-G and original YOLO-G adaptation baselines, which suggests that the idea is not degenerate. However, it does not exceed the ungated multi-scale neck AdvGRL result. The likely interpretation is that multi-scale adaptation is beneficial, but the present objectness gate does not add enough useful selectivity to compensate for the extra weighting bias it introduces. This makes gating a reasonable extension for future refinement rather than the central result of the report.

## 4.7 RainMix Auxiliary Domain

RainMix is not treated as an independent adaptation method. Its role is to supply the auxiliary weather domain required by the triplet formulation. Without the triplet objective, the auxiliary rainy images do not express the intended source--target--auxiliary relationship in the loss. Therefore, RainMix appears in the main methodology as the auxiliary component of the triplet extension, not as a separate paper row.

The rationale is that a synthetic rainy domain may provide a negative weather contrast for target fog features. This is plausible but risky: synthetic rain is not the same shift as fog, and artificial rain artifacts may not match the statistics of Foggy Cityscapes. RainMix is therefore interpreted through the triplet-loss ablation and failure analysis rather than claimed as an independent method.

A second reason for keeping the auxiliary domain explicit is interpretability. If RainMix is only mentioned in a run name or a supplement command, the reader may miss that the negative result is methodologically meaningful. By naming it in the methodology as the auxiliary domain for the triplet objective, the report clarifies that the failure is about a loss design, not about a stray dataset artifact.

## 4.8 RainMix-Coupled Image-Level Triplet Objective

The triplet objective uses source, target, and RainMix auxiliary-domain features to impose a metric-learning relation. The intended relation is that target features should be closer to source features than to auxiliary rainy features. In principle, this can encourage target features to align with the labeled clear-weather source while treating the auxiliary domain as a negative weather contrast.

This objective depends on RainMix because it requires a negative auxiliary distribution. It adds another optimization signal on top of detection and adversarial adaptation losses. That extra signal can be useful if the auxiliary domain is well chosen, but it can also push representations in an unhelpful direction if the negative domain does not reflect the target-domain shift. For this reason, the methods section explains the motivation, while the experiments and discussion determine whether the RainMix-coupled triplet objective is a useful extension or a failure mode.

The triplet objective is therefore evaluated as a hypothesis about representation geometry. The desired geometry is simple: target fog features should move closer to labeled source features and farther from auxiliary rainy features. The risk is that the semantic and low-level dimensions do not separate so cleanly. Fog and rain can both reduce visibility, alter contrast, and add weather-dependent artifacts, while clear-weather source images can still share scene layout and object content with both. If the negative domain is not semantically appropriate, the triplet loss may push the target away from useful weather-robust features rather than toward source-like object representations.

This is why the negative triplet result is reported directly. A failed auxiliary objective is still informative because it identifies a boundary of the method. It shows that the useful part of the evidence is not simply “more domains” or “more losses,” but the specific combination of adaptive adversarial control and multi-scale detector features. The RainMix-coupled triplet result becomes a limitation and an experimental lesson: auxiliary domains require careful validation before they are treated as beneficial regularizers.

## 4.9 Controlled Component-Wise Ablation Design

The experimental design isolates meaningful methodological components while preserving the same detector architecture, data processing, and training entry point. The evaluated components are source-only YOLO-G training, original YOLO-G image-level domain adaptation, adaptive gradient reversal, RainMix-coupled triplet learning, multi-scale neck adaptation, and foreground/objectness-gated adaptation. RainMix is coupled to the triplet objective because that is where it enters the loss formulation as the auxiliary negative domain.

This matters for attribution. If the full component combination underperforms adaptive gradient reversal alone, that is not a contradiction; it means added components changed optimization dynamics or introduced domain mismatch. If the RainMix-coupled triplet objective fails, the component-wise design allows the report to isolate that failure rather than blaming the entire adaptation pipeline. The final ablation table is therefore part of the methodology itself, not merely a reporting convenience.

| Component | Method role | Narrative status |
|---|---|---|
| Source-only YOLO-G training | supervised Cityscapes YOLO-G reference | baseline |
| YOLO-G image-level domain adaptation | DANN-style adversarial feature alignment | original YOLO-G DA baseline |
| Adaptive gradient reversal | dynamic adversarial weighting from classifier state | main contribution |
| RainMix-coupled triplet objective | metric-learning regularizer over source, target, and auxiliary RainMix features | extension/failure analysis |
| Multi-scale neck adaptation | adaptation over P3/P4/P5 representations | extension |
| Objectness-gated adaptation | foreground-weighted spatial alignment | optional extension |

# 5. Experiments

## 5.1 Experimental Setup

The experiments evaluate unsupervised domain adaptation from clear-weather Cityscapes to Foggy Cityscapes. Source-domain images are used with detection labels during training. Target-domain images are used without labels for adaptation and with labels only for validation and final evaluation. All compared methods use the same YOLO-G / YOLOv5-L detector capacity, image resolution, optimization schedule, and data preprocessing so that performance differences can be attributed to the adaptation objective rather than to detector changes.

All compared methods use the same YOLO-G / YOLOv5-L detector capacity, source and target data definitions, image resolution, batch size, and nominal training length. Source-domain images provide detection labels; target-domain images are used without labels during adaptation and with labels only for validation. RainMix images are introduced only in triplet-learning variants, where they serve as the auxiliary negative domain. The reported results are single-run ablations under this protocol unless repeated-seed evidence is explicitly stated.

Keeping the batch size, image size, training length, and data streams fixed is important for domain-adaptive detection because the supervised detection gradient and the adversarial domain-classification gradient both depend on batch composition. Under this protocol, performance differences are attributed to the adaptation objective and feature placement rather than to changes in detector capacity or data loading.

## 5.2 Metrics

The primary detection metrics are mean average precision at IoU threshold 0.5, mean average precision averaged over IoU thresholds from 0.5 to 0.95, precision, recall, and per-class average precision where relevant. The main quantitative table reports target-domain detection performance on Foggy Cityscapes.

Mean average precision at 0.5 IoU is useful because it is sensitive to whether objects are detected at a practically acceptable localization threshold. The stricter averaged metric adds information about localization quality across thresholds and prevents the report from relying only on a lenient criterion. Precision and recall clarify how a method changes the detector's behavior: a method may improve recall by finding more objects while reducing precision, or it may preserve precision while missing difficult fog-obscured objects. The best method is therefore interpreted through the joint metric pattern, not a single number in isolation.

Training diagnostics are also necessary because adversarial adaptation can fail through instability even when final metrics are unavailable or misleading. The diagnostic curves track supervised detection losses, image-level adaptation loss, triplet loss when applicable, detached domain-classifier loss $L_c$, adaptive reversal strength $\lambda_{\mathrm{adv}}$, and run-completion status. A method that produces NaNs, collapses to near-zero detection accuracy, or diverges is reported as a failure rather than omitted.

The diagnostic metrics play a different role from validation metrics. Validation metrics answer whether the final detector works on the target domain. Diagnostic metrics explain how the adaptation objective behaved while training. This distinction is especially important for AdvGRL because the method is explicitly a control rule over adversarial pressure. Without the adaptive-weight and classifier-loss curves, the report could state that the final result improved but could not show whether the proposed mechanism was active or stable.

## 5.3 Ablation Matrix

The ablation study isolates the contribution of each meaningful adaptation component. The source-only YOLO-G detector is the reference. Original YOLO-G image-level domain adaptation tests whether ordinary adversarial feature alignment helps under this domain shift. Adaptive gradient reversal isolates the main proposed training-dynamics change. RainMix-coupled triplet learning tests whether an auxiliary synthetic weather domain is useful when it participates in the triplet loss formulation. The full component combination tests whether adding the triplet extension to adaptive gradient reversal is better than the isolated main method.

```{=latex}
\begin{table}[!t]
\centering
\caption{Controlled ablation matrix. Each row changes one method role or extension relative to the shared YOLO-G detector protocol.}
\label{tab:ablation_matrix}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llp{0.50\textwidth}}
\toprule
ID & Method & Purpose \\
\midrule
A1 & Source-only YOLO-G & clear-weather supervised baseline \\
A2 & YOLO-G DA & fixed image-level adversarial alignment \\
A3 & AdvGRL & adaptive gradient reversal in isolation \\
A4 & Triplet & auxiliary-domain metric-learning extension \\
A5 & AdvGRL + Triplet & combined adaptive reversal and triplet extension \\
A6 & AdvGRL + Neck & multi-scale neck adaptation \\
A7 & AdvGRL + ObjGate & foreground-gated multi-scale adaptation \\
\bottomrule
\end{tabular}
\end{table}
```

These rows are described by method name. Exact execution commands and implementation identifiers remain in the supplement so the manuscript reads like a conference paper while remaining reproducible.

## 5.4 Main Results

The aggregate results table summarizes the full-length ablations from the archived training runs. The RainMix-coupled triplet-only row allows the triplet extension to be evaluated directly rather than inferred only from the full component combination. These aggregate rows follow the main run-summary convention used throughout the report; the later per-class tables evaluate the selected model from each method and therefore should be interpreted as a complementary view rather than as a replacement for this run-summary table.


```{=latex}
\begin{table}[!t]
\centering
\caption{Aggregate detection results on Foggy Cityscapes. The best value in each metric column is shown in bold.}
\label{tab:aggregate_results}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{llcccc}
\toprule
ID & Method & mAP@0.5 & mAP@0.5:0.95 & Precision & Recall \\
\midrule
A1 & Source-only & 0.37841 & 0.24214 & 0.74702 & 0.32365 \\
A2 & YOLO-G DA & 0.41195 & 0.25884 & \textbf{0.77726} & 0.35936 \\
A3 & AdvGRL & 0.39839 & 0.25088 & 0.67457 & 0.35689 \\
A4 & Triplet & 0.31183 & 0.18733 & 0.73212 & 0.27307 \\
A5 & AdvGRL + Triplet & 0.34015 & 0.21109 & 0.74506 & 0.28123 \\
A6 & AdvGRL + Neck & \textbf{0.46665} & \textbf{0.28955} & 0.72261 & \textbf{0.41646} \\
A7 & AdvGRL + ObjGate & 0.44293 & 0.27584 & 0.74389 & 0.38448 \\
\bottomrule
\end{tabular}
\end{table}
```


Figure 1 visualizes the same ablation evidence. The main pattern is that YOLO-G image-level domain adaptation improves over the source-only YOLO-G detector, RainMix-coupled triplet learning degrades target-domain detection, and the strongest result comes from adaptive gradient reversal applied to multi-scale neck features.

![Main ablation results on Foggy Cityscapes. The bars report target-domain detection performance for each completed method row. YOLO-G image-level domain adaptation improves over the source-only YOLO-G baseline, while multi-scale neck adaptive gradient reversal gives the strongest evaluated result.](figures/fig_ablation_main.pdf){#fig:ablation-main width=95%}

The experimental evidence shows that YOLO-G image-level domain adaptation improves over the source-only YOLO-G detector in this run set. The isolated adaptive-gradient row does not exceed the original YOLO-G adaptation row on mAP, but it remains central to the methodology because it directly tests adversarial-pressure control rather than simply adding more losses. This distinction matters for the report narrative: the method contribution is about regulating adversarial training, while the empirical conclusion follows the completed ablation evidence.

The isolated RainMix-coupled triplet adaptation underperforms the source-only YOLO-G detector, YOLO-G image-level domain adaptation, and isolated adaptive gradient reversal. The full RainMix-coupled combination also underperforms both the original YOLO-G adaptation baseline and isolated adaptive gradient reversal. Together, these results support the failure-analysis narrative: adding the auxiliary triplet signal introduces representation pressure that does not improve fog-domain detection in this setting.

The best-performing evaluated configuration is multi-scale neck AdvGRL. It reaches 0.46665 mAP@0.5 and 0.28955 mAP@0.5:0.95, with precision 0.72261 and recall 0.41646. Relative to the source-only YOLO-G detector, this is an absolute improvement of 0.08824 mAP@0.5. Relative to fixed image-level YOLO-G adaptation, the gain is 0.05470 mAP@0.5.

The precision and recall pattern adds another interpretation. The best multi-scale neck setting has lower precision than the YOLO-G image-level domain-adaptation row, but it has substantially higher recall. In fog-domain detection, this suggests that the strongest method finds more target-domain objects rather than only making the detector more conservative. The accompanying mAP@0.5:0.95 gain indicates that this recall improvement is not merely produced by loose detections at the easiest overlap threshold. The method improves the target-domain ranking and localization behavior enough to raise both reported mAP measures.

The foreground-gated extension reaches 0.44293 mAP@0.5, which is also above the source-only YOLO-G and original YOLO-G adaptation baselines. However, it remains below the ungated multi-scale neck setting. This result prevents an overly broad conclusion that foreground weighting is always beneficial. Instead, the evidence supports a narrower claim: multi-scale neck adaptation with adaptive adversarial weighting is beneficial in the experiments, while the present objectness gate is a plausible but not superior refinement.

Per-class evaluation sharpens that interpretation, but it must be read with the correct provenance. Table~\ref{tab:per_class_ap50} evaluates the selected model from each training run. Because these values come from single-model validation, their mean AP columns can differ slightly from the run-summary values above. The strongest multi-scale neck setting does not win every class, but it produces the highest mean AP@0.5 and the highest mean mAP@0.5:0.95 in this model-level view. Its main advantage comes from large improvements on `car` and especially `train`, while `bus` and `truck` remain competitive and the `bicycle`, `person`, and `rider` classes stay close to the fixed image-level baseline. This pattern is consistent with a method that improves broad target-domain robustness without uniformly dominating every class.


```{=latex}
\begin{table}[!t]
\centering
\caption{Per-class AP@0.5 on Foggy Cityscapes. Best values are shown in bold.}
\label{tab:per_class_ap50}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccc}
\toprule
ID & Method & bus & bicycle & car & motorcycle & person & rider & train & truck & mAP \\
\midrule
A1 & Source-only & 0.383 & 0.390 & 0.580 & 0.315 & 0.483 & 0.518 & 0.309 & 0.316 & 0.412 \\
A2 & YOLO-G DA & 0.409 & \textbf{0.450} & 0.601 & \textbf{0.382} & 0.485 & \textbf{0.538} & 0.354 & 0.277 & 0.437 \\
A3 & AdvGRL & 0.415 & 0.437 & 0.585 & 0.338 & 0.483 & 0.514 & 0.279 & 0.312 & 0.420 \\
A4 & Triplet & 0.332 & 0.327 & 0.463 & 0.258 & 0.387 & 0.443 & 0.136 & 0.213 & 0.320 \\
A5 & AdvGRL + Triplet & 0.331 & 0.389 & 0.527 & 0.274 & 0.440 & 0.468 & 0.140 & 0.283 & 0.357 \\
A6 & AdvGRL + Neck & 0.478 & 0.441 & \textbf{0.659} & 0.357 & 0.496 & 0.523 & \textbf{0.479} & 0.300 & \textbf{0.467} \\
A7 & AdvGRL + ObjGate & \textbf{0.501} & 0.411 & 0.651 & 0.363 & \textbf{0.503} & 0.510 & 0.384 & \textbf{0.368} & 0.461 \\
\bottomrule
\end{tabular}%
}
\end{table}
```


Table~\ref{tab:per_class_ap5095} gives a more localization-sensitive view of the same single-model evaluations. The class ordering changes under this metric: the foreground-gated method is strongest for `bus`, `person`, `rider`, and `truck`, whereas the ungated multi-scale neck method stays strongest overall because it preserves the best class balance and retains a much larger `train` advantage. This strengthens the report's main claim. The practical win is not that one adaptive method dominates all classes, but that multi-scale neck adaptation gives the best aggregate fog-domain trade-off across object categories and IoU thresholds.


```{=latex}
\begin{table}[!t]
\centering
\caption{Per-class AP@0.5:0.95 on Foggy Cityscapes. Best values are shown in bold.}
\label{tab:per_class_ap5095}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccc}
\toprule
ID & Method & bus & bicycle & car & motorcycle & person & rider & train & truck & mAP \\
\midrule
A1 & Source-only & 0.324 & 0.204 & 0.405 & 0.171 & 0.276 & 0.318 & 0.145 & 0.257 & 0.263 \\
A2 & YOLO-G DA & 0.337 & \textbf{0.234} & 0.413 & \textbf{0.195} & 0.279 & \textbf{0.326} & 0.156 & 0.216 & 0.269 \\
A3 & AdvGRL & 0.340 & 0.225 & 0.406 & 0.172 & 0.279 & 0.314 & 0.0719 & 0.228 & 0.254 \\
A4 & Triplet & 0.265 & 0.158 & 0.318 & 0.121 & 0.217 & 0.264 & 0.0569 & 0.156 & 0.195 \\
A5 & AdvGRL + Triplet & 0.270 & 0.200 & 0.373 & 0.134 & 0.253 & 0.291 & 0.0696 & 0.208 & 0.225 \\
A6 & AdvGRL + Neck & 0.388 & 0.226 & \textbf{0.439} & 0.186 & 0.279 & 0.316 & \textbf{0.279} & 0.232 & \textbf{0.293} \\
A7 & AdvGRL + ObjGate & \textbf{0.399} & 0.212 & 0.438 & 0.176 & \textbf{0.288} & 0.320 & 0.140 & \textbf{0.276} & 0.281 \\
\bottomrule
\end{tabular}%
}
\end{table}
```


One caution follows from these tables. The apparent `train` gain for A6 should not be overgeneralized because the Foggy Cityscapes validation set contains only 23 `train` instances, whereas `car` and `person` dominate the class counts. The more reliable class-level conclusion is that the winning method improves the major frequent categories while avoiding the severe across-class collapse seen in the RainMix-coupled rows.

```{=latex}
\begin{table}[!t]
\centering
\caption{Comparison of the selected multi-scale neck result, nearby sweep evidence, and foreground-gated extension.}
\label{tab:sweep_context}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llcccc}
\toprule
Category & Method & mAP@0.5 & mAP@0.5:0.95 & Precision & Recall \\
\midrule
Strongest evaluated & AdvGRL + Neck & 0.46665 & 0.28955 & 0.72261 & 0.41646 \\
Nearby sweep point & AdvGRL + Neck & 0.45068 & 0.27661 & 0.76088 & 0.39070 \\
Foreground-gated ext. & AdvGRL + ObjGate & 0.44293 & 0.27584 & 0.74389 & 0.38448 \\
\bottomrule
\end{tabular}
\end{table}
```

Figure 2 shows the completed adaptive-threshold sweep for the multi-scale neck setting. The best-performing point occurs at alpha 0.75. Nearby settings remain competitive but do not exceed it, which supports treating the selected multi-scale neck configuration as the strongest completed result rather than as an arbitrary choice.

![Adaptive-threshold sweep for multi-scale neck adaptive gradient reversal. The alpha 0.75 setting reaches the highest verified mAP@0.5 and mAP@0.5:0.95 among the completed sweep points.](figures/fig_neck_alpha_sweep.pdf){#fig:neck-alpha-sweep width=75%}

Three claims follow from these results. First, the core ablation table evaluates component attribution. Second, the multi-scale neck adaptive-gradient setting is the best-performing run configuration and provides the strongest quantitative evidence for the AdvGRL-centered approach. Third, the RainMix-coupled triplet results are negative in the experimental evidence set, both as an isolated extension and inside the full component combination. More components do not automatically imply better adaptation; the ablation evidence determines the claim strength.

## 5.5 Training Dynamics

Training dynamics are central to this work because the main method changes the adversarial signal over time. Diagnostic plots track the adaptive reversal strength and detached classifier loss over training and relate those curves to target-domain detection performance. These plots show whether the method behaves as intended: measuring classifier state first and then increasing or reducing adversarial pressure accordingly.

Figure 3 compares target-domain detection metrics across training. It shows that final rankings emerge over the full schedule rather than early in training. The multi-scale neck adaptive-gradient setting finishes with the strongest target-domain accuracy among the plotted methods.

![Target-domain detection performance over the full training schedule. Curves compare the source-only YOLO-G detector, YOLO-G image-level domain adaptation, deepest-feature adaptive gradient reversal, multi-scale neck adaptive gradient reversal, and its foreground-gated extension.](figures/fig_training_curves.pdf){#fig:training-curves width=95%}

Figure 4 reports the internal diagnostics recorded for the strongest evaluated setting. These curves expose the adaptive weight, detached classifier probe loss, image-level adaptation loss, and adaptation scale over iterations. They provide evidence about the mechanism beyond the final detection metric and make abrupt instability observable.

![Adaptive-gradient diagnostics for the strongest evaluated multi-scale neck setting. The panels show the adaptive reversal weight, detached classifier probe loss, image-level adaptation loss, and adaptation scale over training iterations.](figures/fig_advgrl_diagnostics.pdf){#fig:advgrl-diagnostics width=95%}

The diagnostic analysis compares YOLO-G image-level domain adaptation and adaptive gradient reversal. Important questions include whether the original YOLO-G adaptation baseline applies excessive pressure early in training, whether adaptive reversal avoids abrupt destabilization, and whether changes in the classifier loss precede changes in detection metrics.

The training-dynamics analysis avoids overinterpreting short sanity runs. Short runs can reveal crashes and boundedness, but final ranking relies on full-length ablations. If platform differences produce inconsistent early behavior, that observation is treated as evidence of instability rather than noise to ignore.

## 5.6 Failure Analysis: RainMix and Triplet

The RainMix-coupled triplet extension is analyzed explicitly because its failure is methodologically informative. Its expected benefit is that an auxiliary synthetic weather domain may provide useful contrast for feature learning. However, synthetic rain is not the same shift as fog. If the auxiliary domain contains artifacts or weather statistics that are poorly matched to Foggy Cityscapes, it can introduce representation pressure that conflicts with the target task.

The triplet objective depends on the assumption that pushing target features away from rainy auxiliary features helps align them with clear-weather source features. That assumption may be false if fog and rain share degradation patterns or if the auxiliary images alter low-level statistics in a way that does not correspond to semantic domain structure. In that case, the triplet loss can push features in an unhelpful direction.

The failure analysis considers four mechanisms: synthetic-domain mismatch, changed batch composition, competition between detection and adaptation objectives, and sensitivity to early adversarial pressure. Because the RainMix-coupled triplet variant underperforms adaptive gradient reversal alone, it is framed as an investigated extension with negative evidence rather than as the central method.

The isolated RainMix-coupled triplet row is especially informative because it performs worse than the source-only YOLO-G detector. This means the auxiliary triplet signal is not merely failing to add a small improvement; it is actively associated with lower target-domain detection quality in the evaluated run. The full combination improves over the isolated triplet row but remains below YOLO-G image-level domain adaptation and below the AdvGRL-family results. That pattern suggests that adaptive adversarial training partially offsets harmful auxiliary pressure but does not turn the triplet extension into a beneficial component.

This negative result is interpreted carefully. It does not prove that all synthetic auxiliary domains are harmful. It does show that this particular auxiliary-domain formulation, under this training protocol and this target fog shift, should not be presented as the central contribution. The strongest conclusion remains AdvGRL-centered and feature-level-specific, while RainMix-coupled triplet learning becomes a documented limitation and a useful direction for future redesign.

## 5.7 Qualitative Results

Qualitative results compare representative Foggy Cityscapes predictions across the source-only YOLO-G detector, YOLO-G image-level domain adaptation, and the strongest evaluated multi-scale neck AdvGRL setting. The figure is used as a visual explanation of the ablation pattern rather than as a substitute for quantitative validation.

The qualitative section supports, but does not replace, the metric evidence. The same image and confidence threshold are used across the displayed methods, so visible changes in the detections can be interpreted alongside the main table and precision--recall plot.

The single-image prediction comparison below uses the same Foggy Cityscapes sample for three models: source-only YOLO-G, YOLO-G image-level domain adaptation, and the best-performing multi-scale neck AdvGRL setting. It is included as a qualitative illustration rather than as an additional metric. The three-panel progression makes the original YOLO-G adaptation baseline visible between source-only training and the strongest evaluated method while keeping the main conclusion tied to the full validation table.

![Qualitative prediction comparison on one Foggy Cityscapes image. All three panels use the same sample and confidence threshold. From left to right, the panels show source-only YOLO-G, YOLO-G image-level domain adaptation, and the best-performing multi-scale neck AdvGRL model.](figures/fig_prediction_comparison.pdf){#fig:prediction-comparison width=100%}

The precision--recall plot provides a compact view of the ablations and helps interpret whether mAP differences are associated with broader recall gains or only with precision shifts.

![Precision--recall trade-off across ablations. Labels identify the ablation rows from the main results table. The best multi-scale neck adaptive-gradient setting improves recall relative to the source-only YOLO-G and original YOLO-G adaptation rows while preserving competitive precision.](figures/fig_precision_recall.pdf){#fig:precision-recall width=75%}

\clearpage

## 5.8 Reproducibility Artifacts

Commands, configurations, metric files, diagnostic logs, and validation outputs are kept in the supplementary material. The paper reports only values that trace to those artifacts, preserving readability while keeping both positive and negative ablation outcomes auditable.

# 6. Discussion

## 6.1 AdvGRL as Training-Dynamics Control

The results indicate that adaptive reversal is most effective when it is paired with the appropriate representation level. Applying AdvGRL only at the deepest backbone representation reaches 0.39839 mAP@0.5, below the YOLO-G image-level domain adaptation baseline. In contrast, applying the same adaptive principle across the multi-scale neck reaches 0.46665 mAP@0.5, the strongest evaluated result. This difference suggests that adversarial-weight control alone is insufficient: the detector features receiving that signal also determine whether alignment benefits target-domain detection.

The multi-scale neck contains representations used directly by the detection head for objects at different spatial scales. Aligning these features can therefore influence fog-domain detection more directly than aligning only one deep representation. The result supports an interpretation of AdvGRL as training-dynamics control whose value depends on where adaptation gradients enter the detector. However, because the evidence consists of completed single-run configurations rather than repeated-seed estimates, this interpretation should be treated as a supported hypothesis rather than a universal claim.

## 6.2 Component-Wise Design as Experimental Design

The component-wise design prevents gains from being incorrectly attributed to the full method. YOLO-G image-level domain adaptation improves over source-only YOLO-G training by 0.03354 mAP@0.5, while the isolated deepest-feature AdvGRL setting does not improve over the original YOLO-G adaptation baseline. The best multi-scale neck AdvGRL setting then improves over the original YOLO-G adaptation baseline by 0.05470. These comparisons show why method names alone are inadequate: adaptive weighting, representation level, auxiliary-domain learning, and foreground gating each alter optimization differently.

Separating these components also exposes negative evidence. The RainMix-coupled triplet objective reduces performance to 0.31183 mAP@0.5 without AdvGRL and 0.34015 when combined with AdvGRL. Likewise, foreground gating reaches 0.44293, which remains above the source-only and original YOLO-G adaptation baselines but below ungated multi-scale neck AdvGRL. Reporting these outcomes preserves causal clarity and avoids presenting every added component as beneficial.

## 6.3 Why the Full Component Combination Is Not the Best Method

The full component combination underperforms the best isolated configuration because additional objectives need not produce complementary gradients. Detection learning, adversarial domain confusion, and triplet metric learning compete over the same representation. When the auxiliary rainy domain does not accurately model the target fog shift, the triplet objective can encourage separation directions that conflict with detection-relevant alignment.

The results therefore distinguish methodological completeness from empirical quality. The full combination is useful as an ablation because it tests the interaction among all originally considered components. It should not be selected as the final method merely because it contains more components. The best-performing evidence instead favors adaptive multi-scale neck alignment without the RainMix-coupled triplet objective or foreground gate.

## 6.4 Practical Lessons

Four practical lessons follow from the experiments. First, adaptation strength and adaptation location must be evaluated jointly; adaptive weighting at one feature level does not guarantee improvement. Second, full-length ablations are necessary because short sanity runs reveal wiring and instability but cannot reliably rank methods. Third, synthetic auxiliary domains require task-specific validation before they are incorporated into metric-learning objectives. Fourth, diagnostic logging of domain-classifier loss, adaptive weight, and adaptation losses is part of the experimental method because it makes collapse and harmful objective interactions observable.

For practitioners, the results argue against treating domain adaptation as a single switch. The same detector can respond differently depending on whether the adaptation signal enters a deep feature, a multi-scale neck representation, or a foreground-weighted spatial map. The strongest setting is not the one with the largest number of components; it is the one where the adaptive adversarial signal is placed at detector features that directly support multi-scale prediction.

For researchers, the results suggest that future work should evaluate adaptive weighting and feature placement together. A paper that compares only fixed and adaptive adversarial weights at one feature level may miss the interaction that dominates the final result. Similarly, a paper that evaluates only the final full component combination may hide the fact that one extension is helpful while another is harmful. The component-wise ablation design is therefore not just an implementation convenience but a necessary part of the scientific argument.

Finally, the report shows the value of preserving negative results. The RainMix-coupled triplet extension was motivated by a plausible representation-learning idea, but the experimental evidence does not support it for this fog-domain task. Reporting that failure improves the final contribution because it narrows the claim to what the evidence actually supports: adaptive multi-scale adversarial alignment, not a general claim that every auxiliary weather objective improves detection.

## 6.5 Evidence Strength and Claim Boundaries

The strongest claim supported by the completed results is that adaptive gradient reversal combined with multi-scale neck adaptation improves target-domain detection over both the source-only YOLO-G detector and YOLO-G image-level domain adaptation in the archived run set. This claim is directly supported by the main ablation table, the alpha sweep, and the training curves. The gain is large enough to matter in the context of the experiment: 0.08824 absolute mAP@0.5 over source-only YOLO-G training and 0.05470 over YOLO-G image-level domain adaptation.

A weaker claim is that adaptive weighting alone is beneficial. The evidence does not support that broad statement because the isolated deepest-feature AdvGRL row is below the YOLO-G image-level domain-adaptation row. The correct interpretation is conditional: adaptive weighting becomes most useful when paired with the appropriate feature level. This distinction should remain visible because it prevents the narrative from overstating the main method.

The weakest claim is that the full component combination is the best method. The evidence contradicts that claim. The full RainMix-coupled combination underperforms the best multi-scale neck result and remains below the original YOLO-G adaptation baseline. The full combination is therefore not selected as the headline method merely because it includes all proposed components. The headline method is the best-supported method, not the most complex one.

We also avoid a state-of-the-art claim. The experiments are single-domain and appear to be single-run ablations rather than repeated-seed estimates. They are sufficient for a controlled final-report conclusion about this project, but not sufficient for a broad benchmark claim against all domain-adaptive detectors. This boundary strengthens the paper because it makes the empirical scope clear.

## 6.6 Implications for Future Domain-Adaptive Detectors

The results suggest that future domain-adaptive detectors should treat feature-level selection as a first-class design choice. Weather shift affects different parts of the representation hierarchy in different ways. Low-level contrast and texture changes may influence spatial features, while higher-level semantic features may already be partly stable. A multi-scale detector therefore provides several possible adaptation sites, and the best site may depend on the target domain and the object sizes of interest.

Adaptive weighting should also be studied as a control mechanism rather than as a standalone improvement. The value of AdvGRL is not that it always increases the adversarial signal; it is that it makes the signal depend on the domain-classifier state. Future work could extend this idea by using smoother schedules, per-layer adaptive weights, uncertainty-aware domain classifiers, or separate control rules for object and background regions.

The negative RainMix result points to a broader lesson about auxiliary domains. Synthetic auxiliary data can be useful only if the induced representation geometry matches the target task. A rainy auxiliary domain may be visually plausible but still geometrically wrong for fog adaptation. Future work should therefore validate auxiliary-domain assumptions through feature analysis, not only by adding the auxiliary loss to the final model.

The per-class tables suggest a second lesson. Aggregate mAP improvements can hide uneven class behavior, especially in urban-driving benchmarks with strong class imbalance. In this study, the winning method remains strongest on average, but some classes are better served by the foreground-gated variant. Future work should therefore not treat per-class AP as an appendix-only diagnostic. For domain-adaptive detection under weather shift, classwise behavior helps determine whether a gain comes from true robustness or from improvements concentrated in a few dominant categories.

The objectness-gated result suggests another direction. Foreground-aware adaptation remains plausible, but the current gate may be too dependent on detector confidence. A future method could combine objectness gates with uncertainty estimates, teacher-student consistency, or class-conditional alignment so that foreground regions are emphasized without discarding useful background context. The present report provides a baseline for that future work by showing that simple gating is competitive but not superior to ungated multi-scale neck adaptation.

## 6.7 Relationship Between Quantitative and Qualitative Evidence

The study is primarily quantitative because the archived metrics provide the strongest evidence. Nevertheless, qualitative detection examples remain useful for explaining how the metrics arise. A recall gain may correspond to recovering small vehicles in dense fog, while a precision loss may correspond to more false positives in low-contrast background regions. Qualitative grids can make those behaviors visible to readers who cannot infer them from mAP alone.

Qualitative evidence should be selected conservatively. The figure should include representative success and failure cases rather than only the best-looking examples. If the multi-scale neck method improves recall, the qualitative section should show whether it recovers objects missed by the baseline. If RainMix-coupled triplet learning harms performance, the section should show whether the harm appears as missed detections, poor localization, or false positives. The visual analysis should therefore serve the same evidence-bounded narrative as the tables.

Until a complete qualitative grid is added, the precision--recall plot helps bridge quantitative and behavioral interpretation. It shows that the best multi-scale neck setting trades some precision for substantially better recall relative to the original YOLO-G adaptation row. That trade-off is consistent with an adaptation method that finds more target-domain objects under fog, while the mAP@0.5:0.95 improvement indicates that the gain is not limited to low-quality detections.

## 6.8 Reproducibility as Part of the Contribution

Reproducibility is unusually important because the method includes both positive and negative results. Positive results require evidence so that the reported gain is credible. Negative results require evidence so that a failed extension is not mistaken for an accidental omission or an undocumented implementation problem. The supplement therefore functions as part of the scientific record rather than as a convenience appendix.

The study follows three reproducibility principles. First, every metric in the main table traces to an archived result file. Second, implementation commands and file-level details stay in the supplement so that the manuscript remains readable as an academic paper. Third, diagnostic logs are preserved for adaptation runs because final mAP alone cannot explain adversarial training behavior.

This structure also supports later revision. If additional fixed neck-all, qualitative, or multi-seed runs are completed, they can be added to the supplement and then promoted into the paper only when their artifacts are complete. Evidence enters the main narrative only after it can be traced.

The ablation record is part of the contribution because it identifies which components helped, which failed, and which remain optional. Domain-adaptive detection methods often fail through interactions among losses, feature levels, and training schedules rather than through one isolated implementation error; preserving diagnostic artifacts makes those interactions traceable.

# 7. Limitations

## 7.1 Domain Scope

The main domain pair is Cityscapes to Foggy Cityscapes. This is an appropriate test bed for weather-domain adaptation, but it does not cover every real driving condition. A method that improves clear-to-fog adaptation may behave differently under night scenes, snow, rain, low-resolution cameras, or a different city distribution. The study therefore avoids claiming general weather robustness beyond the studied setting.

The single-domain scope also affects the interpretation of RainMix. The auxiliary rainy domain is evaluated only as a negative domain for fog adaptation. Its failure here does not mean rainy auxiliary data can never help. It means that this RainMix-coupled triplet formulation did not help this fog-domain detection problem under the completed training protocol.

## 7.2 Seed and Compute Constraints

The ablations are treated as verified runs, but they should not be interpreted as repeated-seed confidence intervals. Deep detector training and adversarial adaptation can be sensitive to initialization, data order, hardware kernels, and platform differences. Multi-seed evaluation would be necessary before making a stronger statistical claim about effect size.

Compute budget also limits the breadth of hyperparameter search. The alpha sweep provides useful evidence around multi-scale neck AdvGRL, but it does not exhaust all possible base reversal weights, thresholds, or feature combinations. The reported best setting is therefore the best-performing setting in the evaluated run set, not a globally optimized configuration.

## 7.3 Auxiliary-Domain Limitations

RainMix is synthetic and may not represent the target fog domain. The triplet formulation assumes that target fog features should be closer to source clear-weather features than to auxiliary rainy features. That assumption is plausible but not guaranteed. If the auxiliary domain differs in the wrong dimensions, the triplet loss can introduce harmful representation pressure.

The study does not include a full feature-space analysis of RainMix, source, and target clusters. Such an analysis could clarify whether the auxiliary domain is geometrically suitable for triplet learning. Without it, the negative RainMix result is interpreted empirically: the extension underperforms in the ablations, but the precise feature-space mechanism remains a topic for future work.

## 7.4 Diagnostic and Artifact Limitations

Short sanity runs are useful for detecting crashes, NaNs, or wiring errors, but they are not reliable for final method ranking. The main conclusions therefore rely on full-length ablations where available. Optional rows without full artifacts should remain marked as optional rather than being used to support the central claim.

The study also depends on archived metric logs. This is stronger than relying on memory or screenshots, but it still requires careful artifact management. A complete final package should preserve environment metadata, validation logs, and figure-generation scripts so that every reported number can be reproduced or audited.

## 7.5 Claim Boundaries

No state-of-the-art claim is made because the study is single-domain and reports single-run ablations. The supported claim is narrower and more defensible: adaptive gradient reversal with multi-scale neck adaptation is the strongest evaluated method in this project and improves over the source-only and YOLO-G image-level domain adaptation baselines on Foggy Cityscapes.

This boundary is intentional. It keeps the report aligned with the evidence and avoids overstating the contribution. The value of the work is the controlled ablation record and the AdvGRL-centered analysis, including both positive and negative component findings.

The limitation does not weaken the main result; it defines the conditions under which the result should be interpreted. Within the completed Cityscapes-to-Foggy-Cityscapes evidence, adaptive multi-scale neck alignment is the strongest evaluated method. Outside that evidence, the method remains a promising hypothesis that requires broader validation.

# 8. Conclusion

This study asked whether adaptive adversarial weighting can improve or stabilize YOLO-G domain adaptation from clear-weather Cityscapes to Foggy Cityscapes. The experimental evidence supports a qualified answer. YOLO-G image-level domain adaptation improves over the source-only YOLO-G detector, but the strongest result comes from adaptive gradient reversal applied to multi-scale neck features. That setting reaches 0.46665 mAP@0.5, improving by 0.08824 over source-only YOLO-G training and by 0.05470 over YOLO-G image-level domain adaptation.

The evidence also shows that more components do not automatically improve domain adaptation. The RainMix-coupled triplet extension performs poorly as an isolated extension and remains below the fixed-adaptation baseline when combined with AdvGRL. The foreground-gated neck extension is stronger than the baseline rows but does not exceed ungated multi-scale neck AdvGRL. The central claim should therefore focus on adaptive multi-scale adversarial alignment, not on the full component combination.

Future work should evaluate this conclusion with multiple random seeds, additional weather and domain pairs, stronger auxiliary-domain generation, and more systematic schedules for adversarial weighting. This study provides a reproducible ablation framework and an evidence-bounded conclusion: AdvGRL is useful when paired with the right detector features, while RainMix-coupled triplet learning is a negative or limited extension in the studied fog-domain setting.

The final takeaway is methodological rather than merely numerical. Adaptive adversarial rules should be evaluated together with feature placement, diagnostic behavior, and component interactions. The best result is not the largest collection of losses, but the configuration whose training signal matches the detector features most relevant to target-domain prediction. Detection metrics identify the best configuration, while diagnostic curves show whether the adversarial mechanism was active, bounded, and interpretable during optimization.


# 9. Declarations and Reproducibility Statement

## 9.1 Data Availability and Code Availability Statement

The experiments use labeled clear-weather Cityscapes for source supervision, Foggy Cityscapes for unlabeled target-domain adaptation and target validation, and RainMix images only for the auxiliary triplet variants. The repository does not redistribute the restricted Cityscapes-family datasets. Reproduction therefore requires obtaining the datasets through the original dataset providers and configuring equivalent paths.

The implementation artifacts needed to reproduce the reported methods are preserved in the supplementary material. Training and validation scripts, model definitions, gradient-reversal utilities, domain-classifier modules, and auxiliary loss implementations are included. The per-class validation table is generated from saved model weights using automated evaluation scripts.

## 9.2 Artifact Traceability

Every aggregate metric in the main ablation table traces to archived training logs. Every per-class value traces to validation logs from the selected model for each method. The two evidence streams are intentionally distinguished: the aggregate table records the run-summary comparison, while the per-class tables record single-model validation behavior. The final report package includes the Markdown source, generated LaTeX, PDF, bibliography, figure scripts, figure files, and supporting tables so that the written claims can be audited against the saved artifacts.

The main run identifiers are intentionally kept in the supplement and appendix rather than overloaded into the narrative prose. This follows the style of domain-adaptive detection papers, where the main text emphasizes the method family and ablation role while the supplement records exact implementation details, command lines, and model provenance.

## 9.3 Ethics and Dataset Licensing

The study uses public driving-scene datasets and synthetic adverse-weather data for object-detection research. It does not introduce human-subject experiments, user tracking, or personal-data collection beyond the source datasets' existing license terms. The most important ethical constraint is dataset compliance: Cityscapes-family data should be downloaded and used according to the dataset owners' access rules, and any external release of the project should avoid redistributing restricted images or annotations.

Because the application domain is autonomous-driving perception, the results should not be interpreted as deployment-ready safety evidence. The experiments evaluate target-domain validation metrics under a controlled benchmark protocol. They do not prove robustness to all fog densities, sensor types, geographic locations, or safety-critical edge cases. A deployed system would require broader validation, failure-case analysis, calibration, uncertainty monitoring, and safety engineering beyond the scope of this final report.

## 9.4 Author Contributions (CRediT) and AI Assistance Disclosure

Using CRediT terminology, the project author is responsible for conceptualization, methodology, software direction, validation oversight, investigation, data curation, writing, review, and final approval. AI-assisted coding and writing tools supported artifact inspection, result summarization, script creation, table integration, and prose revision. The author remains responsible for verifying metric values, dataset permissions, final wording, and any submission-specific requirements. No claim in the report should be treated as externally certified beyond the archived artifacts listed above.

## 9.5 Conflict of Interest Statement and Funding Acknowledgment

No funding or conflict-of-interest information was provided. If the report is submitted to a course, workshop, or journal, this statement should be replaced with the required institutional funding and competing-interest declarations.

# References

# Appendix A: Reproducibility Notes

Exact execution commands, configuration paths, and artifact manifests are maintained in the supplementary package rather than the main text. The paper reports methods and results; the supplement preserves implementation-level reproduction details.

# Appendix B: Supporting Tables

## Dataset Summary

| Dataset | Role | Labels used in training? | Image count | Notes |
|---|---|---:|---:|---|
| Cityscapes | source | yes | recorded in supplement | clear-weather driving |
| Foggy Cityscapes | target/eval | no for DA training; yes for val | recorded in supplement | foggy target domain |
| RainMix | auxiliary | no | recorded in supplement | synthetic rainy auxiliary domain |

## Final Ablation Results

Table~\ref{tab:aggregate_results} reports the final ablation values used by the main paper-style analysis.

## Per-Class AP@0.5

The per-class AP@0.5 values are reported in Table~\ref{tab:per_class_ap50} in the main results section.

## Per-Class AP@0.5:0.95

The per-class AP@0.5:0.95 values are reported in Table~\ref{tab:per_class_ap5095} in the main results section.

# Appendix C: Figure Interpretation Guide

This appendix records how each generated figure should be interpreted and makes the connection between archived metrics, plotted evidence, and written claims explicit.

## C.1 Main Ablation Figure

The main ablation figure compares the completed method rows using the two reported mAP metrics from the run-summary evidence stream. It should be interpreted together with the aggregate results table, while the per-class tables provide a separate single-model validation view. The key comparison is not simply which bar is tallest, but which component change explains the movement from one row to the next. Source-only training establishes the unadapted detector. YOLO-G image-level domain adaptation tests whether ordinary adversarial alignment helps. The deepest-feature adaptive-gradient row tests the adaptive weighting mechanism without the later multi-scale neck change. The RainMix-coupled rows test whether the auxiliary triplet extension helps or harms.

The figure supports three claims. First, YOLO-G image-level domain adaptation is beneficial relative to source-only YOLO-G training. Second, the RainMix-coupled triplet extension is negative in the experimental evidence set. Third, the strongest result is multi-scale neck adaptive gradient reversal, not the full component combination. These claims are repeated in the main text because they are central to the report's argument.

## C.2 Alpha Sweep Figure

The alpha sweep figure focuses on the adaptive-gradient threshold for the multi-scale neck setting. It shows that alpha 0.75 is the strongest evaluated point among the completed sweep values. The purpose of this figure is not to claim that alpha 0.75 is globally optimal for every detector or dataset. Its purpose is narrower: it justifies why the report identifies this run as the best-performing configuration in the archived evidence.

Nearby settings remain competitive, which suggests that the method is not dependent on a single isolated accident. However, the experimental evidence still favors the alpha 0.75 setting. Future work should repeat this sweep across multiple seeds before treating the exact threshold as a stable hyperparameter recommendation.

## C.3 Training-Curve Figure

The training-curve figure plots target-domain validation metrics across epochs for representative methods. It helps distinguish final performance from early-training behavior. A method can look promising early and then plateau, or it can improve later after the adaptation signal stabilizes. This is why the report avoids ranking methods from short sanity runs.

The curves also help evaluate whether the final result is consistent with the trajectory. A method that reaches a strong final mAP after a gradual upward trend is easier to trust than a method whose final value is an unexplained spike. The curve does not replace repeated-seed validation, but it provides useful evidence that the reported endpoint belongs to a plausible training trajectory.

## C.4 Diagnostic Figure

The diagnostic figure is specific to the adaptive-gradient mechanism. It records internal quantities that are not visible in the detection metrics: the adaptive reversal weight, the detached classifier probe loss, the image-level adaptation loss, and the adaptation scale. These curves are important because AdvGRL is a training-dynamics method. If the adaptive weight were constant or unstable, the method would not be behaving as intended even if the final metric were acceptable.

The diagnostic figure should therefore be read as mechanism evidence. It answers whether the adaptive control variables were logged and bounded during training. It does not by itself prove that the mechanism caused the mAP gain, but it supports the interpretation that the method was active and traceable.

## C.5 Precision--Recall Figure

The precision--recall figure summarizes how each method changes detector behavior. The best multi-scale neck method improves recall relative to the source-only YOLO-G and original YOLO-G adaptation rows while preserving competitive precision. This helps explain why mAP improves: the detector appears to recover more target-domain objects rather than merely becoming more conservative.

The same figure also shows why no single metric should dominate the report. YOLO-G image-level domain adaptation has strong precision, while the best multi-scale neck method has stronger recall and higher mAP. The final conclusion therefore depends on the full metric pattern rather than only one plotted coordinate.

# Appendix D: Reproducibility Interpretation

The supplement records implementation-level details that are intentionally absent from the main prose. The paper presents the scientific argument: what was tested, what improved, what failed, and what conclusion follows. The supplement preserves the operational record: how each run was launched, where each metric came from, and which artifacts support the plots and tables.

This separation also applies to negative results. The RainMix-coupled triplet rows remain in the report because they explain why the final contribution is AdvGRL-centered rather than full-component-centered. Their commands and artifacts remain in the supplement so that readers can audit the negative evidence instead of treating it as an unsupported assertion.

Together, the paper and supplement provide two complementary views of the study. The paper communicates methods and conclusions without source-code identifiers, while the supplement preserves commands, configuration details, run-level evidence, and figure provenance needed for reproduction.
