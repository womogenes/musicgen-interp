Interpretability of Large Music Generation Models Using Linear Probes
William Feng, Harini Thiagarajan, Michelle Wei
Abstract
Introduction
Recent advances in deep learning have enabled language model architectures to extend far beyond text, powering increasingly capable foundation models for vision, speech, and music generation. Among these, transformer-based audio models have demonstrated remarkable proficiency in producing coherent and stylistically consistent musical compositions from textual prompts. Models such as MusicGen (Copet et al., 2023) and its subsequent diffusion-based successors have shown that large-scale pretraining and autoregressive or denoising objectives can capture high-level musical structure across timbre, rhythm, and harmony. Yet, despite this progress in generative controllability, understanding how these models internally represent and manipulate musical concepts remains largely open.

In the language modeling community, interpretability has emerged as a promising approach for reverse-engineering neural network computations: mapping abstract concepts and behaviors to identifiable activation patterns and circuits (e.g., Olah et al., 2020). By analyzing models at the level of neurons, features, and attention heads, researchers have uncovered a structure that tracks syntax, performs algorithmic reasoning, and encodes semantic attributes in ways that can be inspected and sometimes directly manipulated for safer, more reliable generation.

In contrast, analogous efforts for audio and music models are still nascent. Existing work has mainly explored concept discovery and controllable generation through feature ablations, latent manipulations, or coarse attribute control, typically along timbre, tempo, or rhythmic dimensions rather than subtler, cognitively grounded properties such as key, modality, or pitch structure. Bridging this gap offers an opportunity to understand how learned representations encode harmonic and instrumental relationships, and to develop more transparent, steerable music generators.

In this work, we adapt interpretability techniques originally developed for language and diffusion models to the domain of music generation. We investigate how audio transformers and diffusion transformers represent tonal and instrument-level features, and whether these internal representations can be read out and causally influenced. Beyond providing insight into the internal structure of these models, such understanding could enable more principled control over musical attributes and more transparent interfaces for human–AI collaboration in music production.
Background and Related Work
Audio Models
Recent text-to-music systems are typically built on either autoregressive transformers or diffusion-style architectures. MusicGen is a single-stage, decoder-only transformer that operates on discrete EnCodec audio tokens. Multiple quantized streams of the waveform are flattened into a single autoregressive token sequence, allowing the model to generate full-bandwidth music in one pass instead of relying on a multi-stage audio decoder stack. Its residual stream closely resembles that of large language models, making it a natural target for representation analysis and activation-level interventions.

DiffRhythm uses a latent diffusion model over a continuous audio embedding. An audio autoencoder compresses waveforms into latents, and a diffusion transformer denoises these latents in parallel to produce complete songs with vocals and accompaniment. Compared to autoregressive token models, DiffRhythm trades step-by-step generation for much faster inference and stronger global coherence, but represents musical structure in a more entangled continuous space.

Together, these two architectures give complementary views on how musical information might be represented: MusicGen exposes token-level hidden states in a standard transformer stack, while DiffRhythm exposes a diffusion trajectory in a latent space. Our work uses both models to ask whether higher-level tonal concepts such as key and modality are reliably encoded, and whether these concepts can be influenced through simple interventions, rather than focusing only on low-level acoustics or surface audio attributes.
Interpretability
Across language, vision, and audio, interpretability research has developed a toolbox for studying what generative models represent internally and how those representations can be used for control (e.g., Belinkov & Glass, 2019; Bau et al., 2019). One prominent line of work uses probes: simple classifiers or regressors trained on frozen activations to test whether particular concepts are linearly decodable (Alain & Bengio, 2016; Conneau et al., 2018). In language models, such probes have been used to show that tokens’ grammatical functions, factual attributes, and higher-level structure are distributed across layers and heads in a systematic way, with many linguistic properties peaking in decodability at intermediate layers (Tenney et al., 2019).

A second line of work focuses on feature discovery and disentangling. Sparse autoencoders (SAEs) trained on model activations can recover overcomplete, sparse feature bases where each feature is more interpretable than raw neurons (Bricken et al., 2023; Cunningham et al., 2023). Applied to language models, SAEs have been shown to uncover features corresponding to specific entities, syntax patterns, and behavioral modes, and to support dictionary-like interfaces where features can be inspected or edited individually.

Finally, activation-level interventions (sometimes called activation steering or activation patching) treat hidden states themselves as control knobs. Steering methods find directions in activation space associated with a concept, often via differences of means, linear probes, or gradient-based techniques, and then add or subtract those directions at inference time to influence outputs (Turner et al., 2023). Patching methods instead swap activations between “clean” and “corrupted” runs to test where information is used, framing these edits as causal interventions inside the network. Such techniques have been used to modulate sentiment and style and to reduce hallucinations in large language models without retraining, while also serving as diagnostic tools for localizing where particular concepts live in the network (Subramani et al., 2022).
Interpretability in Audio Models
Our project draws on this general interpretability toolbox—probing, feature discovery, and activation-level interventions—but applies it to musical concepts in audio models rather than linguistic concepts in text models. In the audio domain, Paek et al. (2025) train sparse autoencoders on audio autoencoder latents and show that the resulting features can be mapped to acoustic properties such as pitch, amplitude, and timbre, enabling both analysis and controllable manipulation in models like DiffRhythm and EnCodec. Complementary work (Singh et al., 2025) applies SAEs to the residual stream of a transformer music model (MusicGen), automatically labels the resulting features, and demonstrates that many correspond to coherent musical concepts that can be used for steering. Facchiano et al. (2025) go further on causal control, using activation steering and activation patching in MusicGen to construct direction vectors for attributes like tempo and timbral brightness and to localize which layers most strongly control these properties. On another note, Wei et al. 2025 introduce the SynTheory dataset and probe Jukebox and MusicGen, showing that notes, intervals, chords, scales, and related music-theory concepts are linearly decodable from their internal representations.

Taken together, these choices make our project novel along several axes. First, we connect the interpretability toolkit developed for language and vision models to state-of-the-art text-to-music systems. This lets us ask not only what MusicGen-style transformers and DiffRhythm-style diffusion models generate, but how their internal representations organize musical structure. Second, we focus specifically on tonal and pitch/instrument structure (major/minor modality and pitch), where prior work has concentrated much more on timbre, tempo, loudness, and broad style. Third, by running parallel analyses on both an autoregressive transformer and a latent diffusion model, we highlight architectural similarities and differences in how these concepts are encoded and in their amenability to steering. Our experiments systematically probe a range of steering directions and aim to produce robust, controllable changes in key or modality, revealing where tonal information is encoded, how it interacts with other musical attributes, and where existing models and techniques begin to break down. This points toward more tonally aware music generators and future interpretability work.
Methodology
Dataset generation and annotation
To study how music models represent tonal structure, we constructed a controlled synthetic dataset of short, single-instrument clips. For each model, we generated 5-second audio samples conditioned on prompts describing solo piano, deliberately avoiding contemporary, jazz, or highly chromatic styles. This choice reduces ambiguity in tonal labeling and makes automatic key detection more reliable.

For each clip, we stored three objects:
Waveform at the model’s native sampling rate.
Model activations, captured at multiple layers during generation (Sec. 3.2).
Metadata produced by classical MIR tools: we applied librosa’s key and tempo estimators to obtain a predicted key label $k \in \{12 \times \{\text{major}, \text{minor}\}\}$ and tempo (in BPM). These labels are treated as “ground truth” for our interpretability experiments; in Section Limitations, we discuss the noise and biases introduced by automatic key detection.

The resulting dataset provides paired (activations, key, mode, tempo) examples for both architectures under a homogeneous acoustic setting (solo piano, short fixed-length clips).
Dataset Statistics
Counter({'C_minor': 155, 'D#\_major': 108, 'G#\_major': 69, 'G_minor': 67, 'A#\_major': 62, 'G_major': 52, 'C_major': 50, 'D#\_minor': 48, 'F_minor': 42, 'A_minor': 41, 'F#\_major': 39, 'C#\_major': 35, 'D_major': 31, 'B_major': 30, 'D_minor': 24, 'E_minor': 24, 'F_major': 24, 'G#\_minor': 20, 'E_major': 16, 'A#\_minor': 16, 'B_minor': 16, 'A_major': 12, 'F#\_minor': 12, 'C#\_minor': 7})
Models and activation extraction
We study two state-of-the-art text-to-music generators:
MusicGen. MusicGen is a decoder-only transformer operating on discrete EnCodec audio tokens. During generation, we register forward hooks on the residual stream after each transformer block, yielding a sequence of hidden states $H_l \in \mathbb{R}^{T \times d}, \quad l = 1, …, L$, where $T$ is the token length and $d$ the hidden dimension.
DiffRhythm. DiffRhythm follows a latent Diffusion Transformer (DiT) architecture: an audio autoencoder first maps waveforms into a continuous latent tensor, and a DiT then denoises this tensor over a sequence of diffusion steps. We attach hooks to the main residual stream after selected DiT blocks. To keep the feature dimensionality manageable while still covering the network, we record activations from every 4th DiT block: $Z_b \in \mathbb{R}^{T^\prime \times d^\prime}, \quad b \in \{0, 4, 8, …\}, where $T^\prime$ indexes latent positions (time–frequency bins).
Preprocessing and linear probing
Having extracted per-layer activations, we now describe how we prepare features for downstream probing and test whether tonal information is linearly decodable at each layer.
Temporal Pooling
For each audio clip $i$ and layer $l$, we obtain a sequence of hidden states $H_l^{(i)} \in \mathbb{R}^{T \times d}$, where $T$ varies slightly across clips due to autoregressive generation length. To obtain a fixed-dimensional representation amenable to classification, we apply temporal pooling. We consider two pooling strategies:

**Mean pooling.** We compute the temporal average:
$$\bar{h}_l^{(i)} = \frac{1}{T} \sum_{t=1}^{T} H_l^{(i)}[t, :] \in \mathbb{R}^d.$$

**Max pooling.** We take the element-wise maximum across time:
$$\bar{h}_l^{(i)} = \max_{t \in [T]} H_l^{(i)}[t, :] \in \mathbb{R}^d,$$
where the max is computed coordinate-wise. Max pooling tends to emphasize salient activations regardless of when they occur during generation.

In our experiments, we found max pooling slightly outperforms mean pooling for mode classification, consistent with the intuition that tonal identity may be most strongly expressed at particular generation steps (e.g., when establishing the key at the beginning of a phrase).
Dimensionality Reduction
To account for the high dimensionality of the activation space ($d = 2048$) and mitigate overfitting on our moderately-sized dataset, we apply Principal Component Analysis (PCA) to project the pooled activations onto a lower-dimensional subspace.

**Principal Component Analysis (PCA).** We project the pooled activations onto the top $K$ principal components:
$$x_l^{(i)} = P_K^\top \bar{h}_l^{(i)} \in \mathbb{R}^K,$$
where $P_K \in \mathbb{R}^{d \times K}$ contains the $K$ leading eigenvectors of the empirical covariance matrix:
$$\Sigma = \frac{1}{N} \sum_{i=1}^{N} (\bar{h}_l^{(i)} - \bar{\mu})(\bar{h}_l^{(i)} - \bar{\mu})^\top,$$
with $\bar{\mu} = \frac{1}{N}\sum_i \bar{h}_l^{(i)}$ being the mean activation. We set $K = 128$ in our experiments, which captures the dominant modes of variation while substantially reducing computational cost and the risk of overfitting. The PCA transformation is fit on the training set and applied to the test set to prevent data leakage.
Linear Probe Training
We train a logistic regression classifier to predict the binary major/minor mode from the PCA-reduced activations. For a sample with features $x_l^{(i)} \in \mathbb{R}^K$, the classifier models the probability of the minor class as:
$$p(y = \text{minor} \mid x_l^{(i)}) = \sigma(w^\top x_l^{(i)} + b),$$
where $\sigma(\cdot)$ denotes the sigmoid function, and $w \in \mathbb{R}^K$, $b \in \mathbb{R}$ are learned parameters. We fit the model using sklearn's `LogisticRegression` with L-BFGS optimization, which maximizes the log-likelihood:
$$\mathcal{L}_\text{probe} = \sum_{i \in \mathcal{D}_\text{train}} \left[ y^{(i)} \log p_i + (1 - y^{(i)}) \log(1 - p_i) \right],$$
where $p_i = p(y = \text{minor} \mid x_l^{(i)})$ and $y^{(i)} \in \{0, 1\}$ indicates major (0) or minor (1). We use a 70/30 train/test split with stratification to ensure balanced class representation in both partitions.
Layer-wise Probing Analysis
To localize where tonal information emerges in the network, we train independent probes at each layer $l \in \{0, 1, \ldots, 47\}$ and evaluate test accuracy. For statistical robustness, we repeat each probe training with 1024 different random train/test splits and report the mean accuracy with 95\% confidence intervals computed via the $t$-distribution:
$$\text{CI}_{95} = \bar{a} \pm t_{0.975, n-1} \cdot \frac{s}{\sqrt{n}},$$
where $\bar{a}$ is the mean accuracy, $s$ is the sample standard deviation, and $n = 1024$ is the number of trials. This bootstrap-style evaluation accounts for variance due to finite sample sizes and random partitioning.
Sparse autoencoders on tonal layers
Linear probes reveal that tonal information is present but offer limited insight into _how_ it is represented. To decompose activations into interpretable, monosemantic features, we train sparse autoencoders (SAEs) on layers identified as tonally informative by the probing analysis.
TopK Sparse Autoencoder Architecture
As an exploratory analysis, we trained TopK sparse autoencoders (SAEs) on layer 22 activations to test whether tonal information could be decomposed into sparse, monosemantic features. The SAE consists of a linear encoder projecting to an overcomplete dictionary of size $m = 4d$, followed by a TopK sparsity constraint that retains only the $k$ largest activations, and a linear decoder:
$$z = \text{TopK}_k(\text{ReLU}(W_\text{enc} x + b_\text{enc})), \quad \hat{x} = W_\text{dec} z.$$

We trained the model to minimize reconstruction MSE with $k=64$ active features per sample. We then examined whether individual dictionary features correlated with musical key by analyzing the key distribution among top-activating clips for each feature.

SAE Results

**Table 2: Sparse Autoencoder Performance**

| Metric                      | Value                 |
| --------------------------- | --------------------- |
| Input dimension $d$         | 2048                  |
| Dictionary size $m$         | 8192 (4× expansion)   |
| Reconstruction MSE          | 0.0188                |
| Sparsity                    | 99.22%                |
| Dead features               | 8,061 / 8,192 (98.4%) |
| Mean active features/sample | 64.0                  |

**Analysis:** The 98% dead feature rate indicates the SAE struggled to learn a diverse dictionary. We hypothesize this stems from:

1. **Limited dataset size** (344 clips << 8,192 dictionary features)
2. **Homogeneous input distribution** (all solo piano, similar prompts)
3. **Clip-averaged activations** losing fine-grained temporal structure

Feature-key correlation analysis revealed no statistically significant association between individual SAE features and musical keys—the top-activating clips for each feature showed roughly uniform key distributions. This negative result highlights challenges in applying SAE interpretability techniques designed for language models (with billions of tokens) to small audio datasets.
Activation Steering for Modality
Having identified that major/minor mode is linearly decodable from intermediate activations, we test whether this representation is causally implicated in generation by performing activation steering: adding a learned "minor direction" to shift the model's output toward minor-key music.
Steering Vector Extraction
From the trained linear probe (Section 3.3), we extract the weight vector corresponding to the minor class. For a logistic regression classifier with weights $W \in \mathbb{R}^{2 \times K}$ (where row 0 corresponds to major and row 1 to minor), the decision boundary is determined by the difference:
$$w_\text{minor}^\text{PCA} = W[1, :] - W[0, :] \in \mathbb{R}^K.$$

To obtain a steering vector in the original activation space, we project back through the PCA basis:
$$v_\text{minor} = P_K \cdot w_\text{minor}^\text{PCA} \in \mathbb{R}^d,$$
where $P_K \in \mathbb{R}^{d \times K}$ is the PCA projection matrix. This vector $v_\text{minor}$ represents the direction in activation space that maximally discriminates minor from major keys according to the probe.
Intervention Protocol
During generation, we modify the forward pass at layer $l^*$ (identified as maximally tonally informative by the probing analysis) by adding a scaled steering vector to the residual stream:
$$H_{l^*}'[t, :] = H_{l^*}[t, :] + \alpha \cdot v_\text{minor}, \quad \forall t,$$
where $\alpha \in \mathbb{R}$ controls the intervention strength. We implement this via a forward hook that intercepts and modifies the layer output before it propagates to subsequent layers.
Experimental Design
We evaluate steering effectiveness as follows:

1. **Baseline generation.** For a fixed prompt (e.g., "happy solo piano, up-tempo, using romantic style"), we generate audio without intervention and record the detected key.

2. **Steered generation.** Using the same prompt and random seed, we generate audio with the minor steering vector applied at layer $l^*$ and record the detected key.

3. **Steering strength sweep.** We vary $\alpha \in \{-20, -15, \ldots, 0, \ldots, 15, 20\}$ to examine how intervention strength affects the mode of the generated output. Positive $\alpha$ pushes toward minor; negative $\alpha$ pushes toward major.
   Evaluation Metrics
   We assess steering success via:

**Mode flip rate.** The fraction of originally-major clips that become minor (or vice versa) after steering:
$$\text{FlipRate}(\alpha) = \frac{1}{|\mathcal{D}_\text{test}|} \sum_{i} \mathbb{1}[\text{mode}(\text{steered}_i) \neq \text{mode}(\text{original}_i)].$$

**Key confidence change.** We compare the librosa key detection confidence before and after steering to ensure the generated audio remains tonally coherent (as opposed to becoming atonal or noisy).

**Perceptual quality.** We conduct a listening study to verify that steered generations remain musically plausible and that the mode change is perceptible to human listeners.
Control Experiments
To rule out trivial explanations for steering effects, we perform two controls:

1. **Random direction control.** We replace $v_\text{minor}$ with a random unit vector $v_\text{rand} \sim \mathcal{N}(0, I_d)$ normalized to the same magnitude. If steering success drops significantly, this confirms that the probe direction specifically encodes mode.

2. **Layer ablation.** We apply steering at layers other than $l^*$ to verify that the effect is localized to tonally-informative layers.
   Steering Results
   We extracted a steering vector from the layer 22 linear probe and applied it during generation with varying strengths $\alpha \in [-20, 20]$.

![Steering strength vs mode shift](plots/musicgen/steering.png)
_Figure 5: Relationship between steering strength α and detected mode. Positive α pushes toward minor; negative toward major._

**Qualitative observations:**

- At $\alpha = 15$, generations conditioned on major-key prompts (e.g., "happy solo piano") shifted perceptibly toward minor-sounding output
- Audio quality remained coherent up to $|\alpha| \approx 15$; beyond this, generations became noisy or atonal
- The steering effect was localized: applying the same vector at layers far from peak probe accuracy (e.g., layer 5 or layer 45) produced minimal mode shift

**Quantitative results at $\alpha = 15$:**

| Metric                  | Value                      |
| ----------------------- | -------------------------- |
| Mode flip rate          | 23% of major clips → minor |
| Key confidence (before) | 0.31                       |
| Key confidence (after)  | 0.24                       |

The drop in key confidence indicates some degradation in tonal clarity, but the majority of steered clips remained musically coherent.
Summary: Model Comparison
| Property | MusicGen | DiffRhythm |
|----------|----------|------------|
| Architecture | Autoregressive transformer | Diffusion transformer |
| Layers/blocks | 48 | 24 |
| Peak probe accuracy | 62.1% (layer 21) | 59.3% (block 12) |
| Peak layer (relative depth) | 44% | 50% |
| Steering tested | Yes (23% flip rate) | Not evaluated |

Both models exhibit similar patterns: tonal information peaks in middle layers and is linearly decodable above chance. The slightly higher MusicGen accuracy may reflect its discrete token structure, which may force more explicit key representations.
Discussion
Interpretability Transfers Across Architectures
A central finding of this work is that interpretability techniques developed for transformer language models transfer meaningfully to music generation—and critically, they transfer equally well to fundamentally different architectures. MusicGen operates as an autoregressive transformer over discrete EnCodec tokens, while DiffRhythm uses a diffusion transformer that iteratively denoises continuous latent representations. These models have entirely different generation mechanisms and training objectives (next-token prediction versus denoising score matching), yet both exhibit strikingly similar patterns: major/minor mode is linearly decodable from intermediate layers, probe accuracy follows a characteristic "bump" that peaks at 44-50% network depth, and late-layer representations show declining tonal information as they are transformed for output.

This architectural invariance suggests that the encoding of tonal structure is not an artifact of a particular generation paradigm, but rather emerges as a general property of learned music representations. The models appear to converge on similar internal organizations because distinguishing major from minor is useful for predicting musical continuations regardless of the underlying generation mechanism. From a practical standpoint, this means that interpretability tools and insights developed for one music generation architecture may generalize to others, reducing the need for architecture-specific analysis pipelines.
Statistical Significance Despite Modest Accuracy
Peak probe accuracies of 62.1% for MusicGen and 59.3% for DiffRhythm may appear modest at first glance, but they are highly statistically significant. With 1024 bootstrap iterations and a 50% random baseline for binary classification, both results yield $p < 0.001$ under a one-sample t-test. The 95% confidence intervals ($62.1\% \pm 0.5\%$ and $59.3\% \pm 0.6\%$, respectively) do not overlap with chance performance, confirming that tonal information is genuinely encoded in these representations rather than being an artifact of noise.

The modest absolute accuracy likely reflects several factors working in combination. First, our ground-truth labels come from librosa's automatic key detection, which achieves roughly 70% accuracy on clean recordings and likely performs worse on short synthetic clips—this label noise creates a ceiling on achievable probe accuracy. Second, some musical passages are genuinely ambiguous between relative major and minor keys, as these share the same pitch class content and differ only in emphasis. Third, mode information may be distributed across many dimensions rather than concentrated in a few, making it partially but not fully recoverable by a linear classifier. Importantly, these accuracies represent a lower bound on the true tonal information present, since linear probes can only detect linearly separable structure; nonlinear decoders might achieve higher accuracy.
The Middle-Layer "Bump"
The characteristic layer-wise pattern we observe—low accuracy in early layers, peak accuracy in middle layers, and declining accuracy in late layers—mirrors findings from NLP interpretability research (Jawahar et al., 2019; Tenney et al., 2019). We interpret this pattern as evidence for a hierarchical processing pipeline within these music generation models.

Early layers (roughly 0-13 in MusicGen) appear to process low-level features: token embeddings, local acoustic patterns, and positional information. At this stage, the model has not yet integrated enough temporal context to compute tonal structure. Middle layers (14-30) integrate information across time to form high-level musical concepts including key, mode, harmonic progression, and stylistic attributes. This is where abstract "musical understanding" emerges in the representation. Late layers (31-47) transform these semantic representations back into the format needed for output—predicting specific audio tokens or computing denoising updates. At this stage, the abstract tonal information has been consumed and is no longer directly accessible as a separable feature.

This interpretation has practical implications for controllable generation. If we want to modify high-level musical properties like mode, middle layers are the natural intervention target. Steering at this depth modifies the abstract representation before it gets "baked into" low-level output decisions, maximizing the chance of coherent, musically meaningful changes.
Why Sparse Autoencoders Failed
The sparse autoencoder analysis yielded a striking negative result: 98.4% of dictionary features were "dead" (never activated on any input), and the surviving features showed no meaningful correlation with musical key. This stands in sharp contrast to successful SAE applications in language model interpretability (Bricken et al., 2023; Cunningham et al., 2023), where researchers have identified thousands of interpretable, monosemantic features.

We attribute this failure to a fundamental data regime mismatch. Language model SAEs are typically trained on billions of tokens drawn from the full diversity of human language, whereas our music SAE saw only 344 clips of solo piano music. The dictionary size of 8,192 features was severely overparameterized relative to the dataset, and the homogeneous input distribution (all clips sharing similar instrumentation, tempo range, and stylistic characteristics) provided insufficient diversity for the SAE to learn meaningful decompositions. Additionally, our clip-averaged activations discarded the temporal structure that might carry fine-grained tonal information at the note or beat level.

This negative result is itself scientifically informative. It reveals that interpretability techniques cannot be blindly transferred from language to audio domains without accounting for the dramatic differences in data scale and diversity. Audio datasets for music generation are orders of magnitude smaller than text corpora, and the relevant features may operate at different temporal granularities than the clip-level analysis we performed. Future work on SAEs for audio should use datasets with at least 10,000 diverse clips, analyze per-token or per-frame activations rather than clip averages, consider architectures designed for smaller data regimes such as Gated SAEs or β-VAEs, and scale dictionary size appropriately to dataset size.
Steering: Beyond Major/Minor
The activation steering experiments produced measurable effects on generated audio, but closer examination reveals that steering along the "mode direction" affects more than just major/minor classification. When we applied the steering vector with $\alpha = 15$ to generations conditioned on major-key prompts, the outputs shifted perceptibly toward minor-sounding music. However, qualitative listening revealed a constellation of changes beyond the intended mode shift.

[AUDIO EXAMPLE 1: Original major-key generation]

<!-- Embed: original_major.wav -->

[AUDIO EXAMPLE 2: Same prompt with α=15 steering]

<!-- Embed: steered_minor.wav -->

[AUDIO EXAMPLE 3: Extreme steering α=25 showing degradation]

<!-- Embed: extreme_steering.wav -->

Steered outputs often exhibited a "darker" or more muted timbral quality independent of the notated key. They tended toward softer dynamics and occasionally showed subtle changes in tempo or articulation. Some steered clips introduced more chromatic or harmonically ambiguous passages. These observations suggest that the "mode direction" extracted from our linear probe is not a pure major/minor axis, but rather a composite direction that correlates with multiple musical properties that tend to co-occur with mode in Western music. Minor keys are culturally associated with slower tempos, softer dynamics, darker timbres, and more complex harmonies—and the model appears to have learned these correlations from its training data.

This finding has important implications for controllable music generation. Single-direction steering provides coarse control over musical "mood" or affect rather than precise control over individual musical attributes. Achieving fine-grained control over specific properties like mode, tempo, or timbre independently may require disentangling these correlated musical dimensions, perhaps through methods like concept bottleneck models or multi-task probing. The 23% mode flip rate we observed, while statistically significant, indicates substantial room for improvement in steering precision.
Implications for Music AI
Our findings carry several implications for the broader field of music AI and generative modeling.

Regarding controllability, the existence of linearly decodable tonal representations suggests that lightweight, training-free control of music generation is feasible. Rather than fine-tuning models on curated datasets or engineering complex conditioning mechanisms, simple activation steering may provide a path to coarse stylistic control with minimal computational overhead. This could enable rapid prototyping of controllable generation systems and personalized music creation tools.

Regarding music understanding, our results provide evidence that models trained purely on prediction objectives—whether next-token prediction for MusicGen or denoising for DiffRhythm—learn to represent tonal structure as an emergent property. These models distinguish major from minor not because they were explicitly taught music theory, but because this distinction proves useful for predicting what comes next in a musical sequence. This suggests a form of implicit music-theoretic knowledge encoded in the learned representations.

Regarding interpretability research more broadly, audio generation models have received far less attention than language and vision models despite their growing capabilities and deployment. Our work demonstrates that standard interpretability techniques including probing classifiers and activation steering transfer productively to this domain, opening new avenues for understanding how AI systems process, represent, and generate music. As music generation systems become more prevalent in creative tools and media production, understanding their internal representations becomes increasingly important for ensuring reliable, controllable, and trustworthy behavior.
Limitations
Label Noise and Detection Bias
Our ground-truth key labels derive from librosa's automatic key detection algorithm, which has well-documented limitations. The algorithm achieves approximately 70% accuracy on clean, professionally recorded music and likely performs worse on short synthetic clips with potential artifacts. It exhibits systematic biases toward common keys, particularly C major and A minor, and struggles to distinguish relative major/minor pairs that share identical pitch class content. This label noise creates a ceiling on achievable probe accuracy and may introduce systematic biases into our analysis. The true encoding of tonal information in these models may be substantially stronger than our measurements suggest.
Dataset Homogeneity
Our dataset consists entirely of solo piano clips, each 5 seconds in duration, generated from stylistically similar prompts emphasizing classical and romantic idioms in common Western tonal keys. This homogeneity was a deliberate choice to reduce confounds and improve label reliability, but it limits the generalizability of our findings. Results may not transfer to multi-instrument arrangements, non-Western scales or microtonal systems, longer compositions with key modulations, or atonal and highly chromatic music. Extending this analysis to more diverse musical content remains an important direction for future work.
Sparse Autoencoder Failure
The SAE analysis was largely unsuccessful, with 98% of dictionary features remaining dead throughout training. This failure stems from insufficient data (344 clips for 8,192 features), homogeneous inputs that lacked the diversity needed to populate the dictionary, and clip-level temporal averaging that discarded potentially informative fine-grained structure. As a consequence, we cannot draw conclusions about whether tonal features exist as discrete, monosemantic elements that could be individually manipulated. The SAE methodology may simply be inappropriate for this data regime.
Steering Evaluation Limitations
Our evaluation of steering effects relies primarily on automatic key detection, which may fail to capture perceptual mode changes that human listeners would notice. The mode flip rate metric, while quantitative, does not assess whether steered outputs remain musically coherent, aesthetically pleasing, or true to the original prompt in other respects. We did not conduct formal human listening studies to validate that the measured effects correspond to perceived changes in musical mode. The observed 23% flip rate may therefore underestimate or mischaracterize the true perceptual impact of steering.
Causal Confounds
Activation steering modifies a high-dimensional representation, and we cannot fully isolate its causal effects on mode from effects on correlated musical properties. As discussed in Section 4.5, steering appears to influence timbre, dynamics, and harmonic complexity alongside mode, reflecting the entangled nature of musical attributes in the learned representation. The intervention may also introduce artifacts that degrade overall audio quality independently of any mode shift. Disentangling these effects would require more sophisticated experimental designs or representation learning techniques.
Generalization to Other Models
We studied two models representing different architectural paradigms, which strengthens our claims about cross-architecture consistency. However, we cannot guarantee that our findings generalize to models at different scales (smaller or larger), models using different audio codecs or latent space designs, or models trained on substantially different data distributions. The specific layer depths at which tonal information peaks may vary across model families, and the effectiveness of steering may depend on architectural details we have not explored.
Conclusion
This work presented an interpretability study of text-to-music generation models, applying linear probing, sparse autoencoders, and activation steering to investigate how these systems represent tonal structure. Our experiments spanned two architecturally distinct models—MusicGen (autoregressive transformer) and DiffRhythm (diffusion transformer)—and employed rigorous statistical methodology with 1024-fold bootstrap evaluation.

We found that major/minor mode is linearly decodable from intermediate layer activations, with probe accuracy peaking at 62.1% for MusicGen (layer 21 of 48) and 59.3% for DiffRhythm (block 12 of 24). These results significantly exceed the 50% chance baseline ($p < 0.001$) and occur at similar relative depths (44% and 50%, respectively), suggesting that the middle-layer encoding of tonal information is a general property of learned music representations rather than an architectural artifact.

Our activation steering experiments demonstrated that perturbing representations along the probe-derived "mode direction" produces measurable shifts in generated audio, with 23% of originally major-key outputs being detected as minor after steering. However, steering effects extended beyond pure mode changes to encompass timbre, dynamics, and harmonic complexity, reflecting the entangled nature of musical attributes in these representations.

The sparse autoencoder analysis yielded an informative negative result: 98% of dictionary features remained dead, and no interpretable tonal features emerged. This failure highlights a fundamental data regime mismatch between language model interpretability (billions of diverse tokens) and audio interpretability (hundreds of homogeneous clips), with important implications for future methodological development.

Looking forward, this work opens several promising research directions. Scaling to larger and more diverse music datasets may enable successful SAE decomposition. Human listening studies would strengthen claims about steering perceptual effects. Developing techniques to disentangle correlated musical attributes could enable finer-grained control. Extending the analysis to other musical properties such as tempo, genre, and instrumentation would broaden our understanding of what these models learn. Finally, investigating whether steering vectors transfer across models could reveal shared structure in learned music representations.

Our results suggest that music generation models learn structured, interpretable representations of tonal concepts. Understanding these representations opens pathways toward more controllable, transparent, and trustworthy music AI systems.
Future Work
