# Interpretability of Large Music Generation Models Using Linear Probes

William Feng, Harini Thiagarajan, Michelle Wei

# Introduction

Transformer-based audio models have achieved remarkable proficiency in generating coherent musical compositions from text prompts. MusicGen (Copet et al., 2023)[https://arxiv.org/abs/2306.05284] produces full-bandwidth music through autoregressive token prediction over discrete EnCodec representations, while diffusion-based successors like DiffRhythm[https://arxiv.org/abs/2503.01183] achieve faster inference through parallel denoising of continuous latents. These models capture high-level musical structure across timbre, rhythm, and harmony. Yet despite their generative success, understanding how they internally represent and manipulate musical concepts remains largely open—a gap that limits both scientific understanding of what these models learn and practical controllability for creative applications.

In the language modeling community, interpretability research has successfully mapped abstract concepts to identifiable activation patterns (Olah et al., 2020)[https://distill.pub/2020/circuits/zoom-in/]. Probing classifiers reveal where grammatical and semantic information is encoded, with many linguistic properties peaking in decodability at intermediate layers (Tenney et al., 2019)[https://arxiv.org/abs/1905.05950]. Sparse autoencoders decompose representations into interpretable monosemantic features (Bricken et al., 2023)[https://transformer-circuits.pub/2023/monosemantic-features]. Activation steering enables training-free control over model outputs by adding concept-associated directions to hidden states (Turner et al., 2023)[https://arxiv.org/abs/2308.10248]. These techniques have transformed our understanding of language models, revealing hierarchical processing, localized concept storage, and causal pathways from representation to behavior.

Analogous work on music generation models remains nascent and limited in scope. Paek et al. (2025) train sparse autoencoders on audio autoencoder latents and discover features corresponding to pitch, amplitude, and timbre—low-level acoustic properties rather than higher-order musical structure. Singh et al. (2025) apply SAEs to MusicGen's residual stream and find features for genre, mood, texture, and instrumentation, demonstrating that transformer music models contain rich internal structure analogous to language models. However, their evaluation concentrates on broad stylistic attributes rather than quantitative tonal analysis. Facchiano et al. (2025) demonstrate activation steering for tempo and timbral brightness in MusicGen, showing that causal patching can localize where these controls live in the network—but they deliberately restrict their concept space, leaving open whether more abstract tonal variables can be similarly steered. Wei et al. (2025) provide the closest precedent, showing that notes, intervals, chords, and scales are linearly decodable from MusicGen and Jukebox representations. Their SynTheory dataset and probing results provide strong evidence that music models implicitly encode music-theoretic structure. However, their work is observational—it demonstrates that information is present but does not attempt causal manipulation, nor does it compare architectures to test whether findings generalize.

We address these gaps by investigating whether major/minor mode—a fundamental tonal concept distinguishing the emotional quality of Western music—is encoded and controllable across architecturally distinct models. Unlike prior work that focuses on single architectures or observational findings, we provide the first cross-architecture comparison of tonal representations and the first causal demonstration that probe-identified mode information influences generation.

Our work makes three primary contributions to music AI interpretability. First, we demonstrate that mode is linearly decodable from intermediate activations of both MusicGen and DiffRhythm, with the characteristic "middle-layer bump" occurring at 44–50% network depth in both cases despite radically different generation mechanisms—autoregressive token prediction versus parallel latent denoising. We verify this pattern with $p < 10^{-150}$ statistical significance across 1,024 bootstrap samples, establishing that tonal encoding is a robust, architecturally general phenomenon rather than an artifact of any particular model family.

Second, we show that probe-derived steering vectors causally influence generated mode. At steering strength $\alpha = 15$, we achieve 23% mode flip rates while random directions of equal magnitude produce negligible effects (< 2%). This causal demonstration bridges the gap between observational probing—which shows information is present—and practical controllability, establishing that tonal representations functionally influence model outputs.

Third, we document that sparse autoencoders fail catastrophically in this data regime, with 98% of dictionary features remaining dead and no surviving features showing tonal specificity. This negative result exposes fundamental limitations when transferring language interpretability methods to audio and motivates development of techniques specifically adapted to smaller, more homogeneous audio datasets.

# Background

## Audio Generation Architectures

MusicGen is a single-stage, decoder-only transformer operating on discrete EnCodec audio tokens (Copet et al., 2023)[https://arxiv.org/abs/2306.05284]. The model encodes waveforms into multiple quantized streams that are flattened into a single autoregressive sequence, allowing full-bandwidth generation in one pass without multi-stage decoding. With 48 transformer layers, hidden dimension 2048, and a residual stream architecture closely resembling GPT-style language models, MusicGen is a natural target for interpretability techniques developed for text. The autoregressive structure means that at each generation step, the model has access to all previously generated tokens, building up musical context incrementally.

DiffRhythm follows a latent Diffusion Transformer (DiT) architecture[https://arxiv.org/abs/2503.01183]. An audio autoencoder first compresses waveforms into continuous latent tensors, and a 24-block DiT then denoises these latents in parallel over a fixed number of diffusion steps to produce complete songs with vocals and accompaniment. Compared to autoregressive models, DiffRhythm trades sequential generation for faster inference and potentially stronger global coherence—the model can "see" the entire latent at each denoising step rather than building context token-by-token. However, this parallel structure represents musical information in a more entangled continuous space that may be harder to interpret.

These architectures offer complementary views on how musical information might be represented internally. MusicGen exposes token-level hidden states at each autoregressive step with clear sequential structure, while DiffRhythm exposes intermediate states along the denoising trajectory in a holistic latent space. By analyzing both, we test whether interpretability findings are architecture-specific or reflect general properties of learned music representations.

## Interpretability Methods

Linear probes are simple classifiers—typically logistic regression—trained on frozen model activations to test whether specific concepts are linearly decodable (Alain & Bengio, 2016)[https://arxiv.org/abs/1610.01644]. The key insight is that linear classifiers have limited capacity: they cannot construct complex features through nonlinear combinations, so successful probing indicates that the target concept is represented in a genuinely linearly accessible form. In language models, Conneau et al. (2018)[https://arxiv.org/abs/1805.01070] use probes to show that syntactic information is encoded in intermediate layers, while Tenney et al. (2019)[https://arxiv.org/abs/1905.05950] demonstrate that different linguistic properties peak at different depths, revealing a processing hierarchy from surface features to abstract semantics. A limitation of probing is that it is correlational rather than causal: high accuracy shows information is present but not that it influences model outputs.

Sparse autoencoders (SAEs) address a different question: not whether information is present, but how it is represented. By training overcomplete dictionaries with sparsity constraints—encouraging each input to be reconstructed from only a few dictionary elements—SAEs decompose entangled activation vectors into interpretable feature bases (Bricken et al., 2023)[https://transformer-circuits.pub/2023/monosemantic-features]. In language models, SAEs recover features corresponding to specific entities, syntactic patterns, and behavioral modes (Cunningham et al., 2023)[https://arxiv.org/abs/2309.08600]. Individual features can then be inspected, compared across contexts, or edited to influence model behavior. However, SAE success depends critically on data scale and diversity—language SAEs typically train on billions of diverse tokens. Whether they work in the smaller, more homogeneous data regimes common in audio is an open question.

Activation steering provides causal tests of representation. The core idea is to identify directions in activation space associated with specific concepts—via probe weights, differences of means, or gradient-based methods—and then add scaled versions of these directions to hidden states during inference (Turner et al., 2023)[https://arxiv.org/abs/2308.10248]. If adding a "concept direction" causes the model to exhibit more of that concept in its outputs, this confirms that the identified representation is not merely epiphenomenal but functionally relevant to generation. Subramani et al. (2022)[https://arxiv.org/abs/2205.05124] use similar techniques to modulate sentiment and style in language models without retraining, demonstrating practical value for controllable generation.

# Methodology

## Dataset Construction

We generated 1,000 five-second solo-piano clips using MusicGen-large, conditioned on prompts describing classical and romantic styles while deliberately avoiding jazz, contemporary, or highly chromatic idioms. This homogeneous setting serves two purposes: it simplifies automatic key detection by avoiding ambiguous tonal contexts, and it isolates tonal structure from timbral and textural variation that might confound our analysis. For each clip, we stored three objects: the raw waveform at 32kHz sampling rate, activations from all 48 transformer layers captured via forward hooks during generation, and metadata including key and mode labels obtained from `librosa`'s chromagram-based key detection algorithm.

The resulting dataset contains 528 clips labeled major and 472 labeled minor by the automatic detector, roughly balanced across modalities. The distribution over specific keys was skewed: C minor (155 clips) and D♯ major (108 clips) were most frequent, followed by G♯ major (69), G minor (67), and A♯ major (62). Less common keys like C♯ minor (7 clips) and A major (12 clips) were substantially underrepresented. We use 70/30 stratified train/test splits throughout to preserve the major/minor ratio in both partitions.

We acknowledge that automatic key detection introduces label noise. The `librosa` algorithm achieves approximately 70% accuracy on clean, professionally recorded music and likely performs worse on short synthetic clips that may contain artifacts or ambiguous tonal centers. Some clips are genuinely ambiguous between relative major and minor keys, which share identical pitch class content and differ only in emphasis. This label noise creates a ceiling on achievable probe accuracy—we cannot expect to exceed the reliability of our supervision signal. Our quantitative results should therefore be interpreted as lower bounds on the true amount of tonal information encoded in these models.

## Activation Extraction and Processing

For MusicGen, we registered forward hooks on the residual stream after each of 48 transformer blocks, capturing hidden states $H_\ell^{(i)} \in \mathbb{R}^{T_i \times 2048}$ for each clip $i$ and layer $\ell$. The sequence length $T_i \approx 256$ varies slightly across clips due to autoregressive generation dynamics. For DiffRhythm, we recorded activations from every 4th of 24 DiT blocks—at indices $b \in \{0, 4, 8, 12, 16, 20\}$—to balance network coverage with computational tractability and storage constraints.

To obtain fixed-dimensional representations amenable to classification, we applied temporal pooling across the sequence dimension. We considered both mean pooling ($\bar{h} = \frac{1}{T}\sum_t h_t$) and max pooling ($\bar{h} = \max_t h_t$, computed coordinate-wise). Max pooling outperformed mean pooling in preliminary experiments by 1–2 percentage points, consistent with the intuition that tonal identity may be most strongly expressed at particular generation moments—perhaps when establishing the key at phrase onsets—rather than uniformly across time.

Given the high dimensionality of the pooled activations (2048) and our moderate dataset size (1000 clips), we applied Principal Component Analysis to project onto the top 128 principal components before training probes. This reduces overfitting risk while retaining the dominant modes of variation that likely carry concept-relevant information. Critically, PCA was fit only on training data and then applied to test data to prevent leakage.

## Linear Probing Protocol

We trained logistic regression classifiers at each layer to predict binary major/minor mode from the 128-dimensional PCA-reduced activations. We used scikit-learn's LogisticRegression with L-BFGS optimization, which maximizes cross-entropy loss without explicit regularization. For statistical robustness, we repeated training over 1,024 random train/test splits and computed 95% confidence intervals via the $t$-distribution:

$$\text{CI}_{95} = \bar{a} \pm t_{0.975, n-1} \cdot \frac{s}{\sqrt{n}}$$

where $\bar{a}$ is the mean test accuracy across splits, $s$ is the sample standard deviation, and $n = 1024$. This bootstrap-style evaluation accounts for variance due to finite sample sizes and random partitioning, allowing us to make rigorous claims about statistical significance.

## Sparse Autoencoder Training

We trained a TopK sparse autoencoder on max-pooled, mean-centered layer 22 activations—selected based on probing results as a tonally informative layer. The architecture consisted of a linear encoder $W_\text{enc} \in \mathbb{R}^{8192 \times 2048}$, a TopK sparsity constraint retaining only the $k=64$ largest activations per sample, and a linear decoder $W_\text{dec} \in \mathbb{R}^{2048 \times 8192}$. The 4× overcomplete dictionary (8192 features for 2048-dimensional inputs) follows standard practice in language SAE work. Training minimized reconstruction MSE: $\mathcal{L} = \|x - \hat{x}\|_2^2$.

After training, we analyzed whether individual dictionary features correlated with musical keys by examining the key distribution among the top-20 activating clips for each feature. We also computed point-biserial correlations between feature activations and binary mode labels.

## Activation Steering Protocol

From the logistic regression probe weights $W \in \mathbb{R}^{2 \times 128}$ (with row 0 for major and row 1 for minor), we extracted a steering direction by taking the difference: $w_\text{minor}^\text{PCA} = W[1,:] - W[0,:] \in \mathbb{R}^{128}$. We then projected back to the original activation space through the PCA basis: $v_\text{minor} = P_{128} \cdot w_\text{minor}^\text{PCA} \in \mathbb{R}^{2048}$, where $P_{128}$ contains the top 128 principal components.

During generation, we modified layer 22 activations via a forward hook that added a scaled steering vector to the residual stream:

$$H_{22}'[t,:] = H_{22}[t,:] + \alpha \cdot v_\text{minor}, \quad \forall t$$

We swept steering strength $\alpha$ over the range $[-20, 20]$. Positive $\alpha$ pushes toward minor; negative $\alpha$ pushes toward major. For each steered generation, we re-ran `librosa` key detection to measure the mode of the output and its confidence score.

Two control experiments verified that observed effects were specific to the probe-derived direction rather than generic perturbation artifacts. First, we replaced $v_\text{minor}$ with random unit vectors of the same norm and measured flip rates. Second, we applied the original $v_\text{minor}$ at non-optimal layers (5 and 45, in the early and late portions of the network) where probing showed lower mode information.

# Results

## Layer-wise Probing Reveals Middle-Layer Tonal Encoding

MusicGen probe accuracy rose from near-chance in early layers (layer 0: 56.3%) through the middle of the network to peak at layer 21 with 62.1% ± 0.5% test accuracy over 1,024 splits, then declined toward the output (layer 47: 57.2%). DiffRhythm showed qualitatively identical patterns across its sampled DiT blocks, peaking at block 12 with 59.3% ± 0.6% accuracy.

![Linear probe accuracy across DiffRhythm transformer blocks with 95% confidence intervals. Test accuracy peaks at block 12 (59.3%), corresponding to 50% of network depth, significantly above the 50% random baseline shown as a dashed line.](linearprobdr.png)
_Figure 1: Layer-wise probe accuracy for DiffRhythm demonstrates the characteristic middle-layer bump observed in NLP interpretability. Tonal information peaks at intermediate depth and declines toward the output._

The statistical significance of these results is overwhelming. Computing one-sample $t$-statistics against the 50% null hypothesis:

$$t = \frac{\bar{a} - 0.5}{s/\sqrt{n}}$$

yields $t \approx 75.2$ for MusicGen and $t \approx 48.1$ for DiffRhythm. The corresponding $p$-values are smaller than $10^{-150}$—these are not sampling artifacts or chance fluctuations, but genuine signals that tonal information is encoded in intermediate representations.

| Property                    | MusicGen                   | DiffRhythm            |
| --------------------------- | -------------------------- | --------------------- |
| Architecture                | Autoregressive transformer | Diffusion transformer |
| Total layers/blocks         | 48                         | 24                    |
| Peak test accuracy          | 62.1% ± 0.5%               | 59.3% ± 0.6%          |
| Peak layer (absolute)       | 21                         | 12                    |
| Peak layer (relative depth) | 44%                        | 50%                   |
| $t$-statistic vs. chance    | 75.2                       | 48.1                  |

_Table 1: Cross-architecture comparison reveals remarkably similar tonal encoding patterns despite radically different generation mechanisms and training objectives._

PCA visualization of activations at peak layers reveals partial but incomplete separation between major and minor classes in both models, consistent with the above-chance but imperfect probe accuracy.

![PCA of MusicGen layer 22 activations colored by detected mode (major in blue, minor in red). Partial separation is visible along the first principal component.](pcamode.png)
_Figure 2: PCA visualization shows partial major/minor separation in MusicGen layer 22 activations. The substantial class overlap explains the 62% accuracy ceiling—perfect separation would yield much higher accuracy._

![PCA of DiffRhythm block 12 activations colored by detected mode (major in blue, minor in orange). Similar partial separation pattern as MusicGen.](pcakeydr.png)
_Figure 3: DiffRhythm block 12 shows similar PCA structure to MusicGen, supporting the hypothesis that tonal representations emerge similarly across architectures._

## Sparse Autoencoder Analysis: An Informative Failure

The SAE achieved reconstruction MSE of 0.0188 with 99.22% sparsity—the TopK constraint successfully enforced that only 64 of 8192 features activate per sample. However, 8,061 of 8,192 dictionary features (98.4%) remained "dead"—they never activated on any input across the entire dataset.

| Metric                        | Value                  |
| ----------------------------- | ---------------------- |
| Dictionary size               | 8,192 (4× expansion)   |
| Active features per sample    | 64 (TopK constraint)   |
| Dead features                 | 8,061 (98.4%)          |
| Reconstruction MSE            | 0.0188                 |
| Mean feature-mode correlation | 0.02 (not significant) |

_Table 2: SAE statistics reveal catastrophic underutilization of dictionary capacity. Nearly all features remain dead, and surviving features show no tonal specificity._

For the 131 surviving active features, we examined the key distribution among their top-20 activating clips. No feature showed statistically significant association with specific keys or modes—the distributions closely matched the global dataset distribution, indicating that these features capture general activation patterns rather than tonal content.

This failure reflects a fundamental data regime mismatch rather than a problem with our implementation. Language SAEs train on billions of diverse tokens from the full distribution of human text; our SAE saw only 344 usable clips (after filtering low-variance samples) from a narrow distribution of solo piano music. The 8,192-feature dictionary was overparameterized by roughly 20× relative to our data, and the homogeneous input distribution provided insufficient diversity to populate the dictionary with distinct, meaningful features.

This negative result is scientifically valuable: it establishes that interpretability techniques cannot be naively transferred from text to audio without adapting to the dramatic differences in data scale and diversity. Future audio SAE work will likely require either much larger and more varied datasets, or architectures specifically designed for small-data regimes such as gated SAEs or β-VAEs.

## Activation Steering Demonstrates Causal Influence

At steering strength $\alpha = 15$, 23% of clips originally detected as major were detected as minor after intervention—a substantial mode flip rate. The mean key detection confidence dropped from 0.31 (baseline) to 0.24 (steered), indicating some degradation in tonal clarity but not collapse into atonality. For $|\alpha| > 15$, artifacts increased substantially, with generations becoming noisy or incoherent.

Control experiments confirmed that these effects are specific to the probe-derived direction and localized to tonally informative layers. Replacing $v_\text{minor}$ with random unit vectors of the same norm produced flip rates below 2% across all $\alpha$ values tested—statistically indistinguishable from zero. Applying the original $v_\text{minor}$ at layer 5 (early network) or layer 45 (late network) produced flip rates below 5%, far less than the 23% observed at layer 22.

![Sorted steering vector coefficients across the 2048 dimensions of layer 22 activations. The distribution is highly non-uniform with a few large negative and positive weights.](steering.png)
_Figure 4: Steering vector weight distribution shows sparse structure. Mode information is concentrated in specific activation dimensions rather than uniformly distributed, consistent with the success of linear probing._

**Audio Examples:** Steering effects across different $\alpha$ values. Each clip shows the progression from major-biased (α=-15) through baseline (α=0) to minor-biased (α=+15).

<details class="my-4">
<summary class="cursor-pointer font-semibold">Clip 0034</summary>
<table class="w-full my-2">
<tr><td class="py-1">alpha = -15 (toward major)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0034/alpha_-15.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = -02</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0034/alpha_-02.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +00 (baseline)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0034/alpha_+00.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +02</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0034/alpha_+02.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +15 (toward minor)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0034/alpha_+15.wav" type="audio/wav"></audio></td></tr>
</table>
</details>

<details class="my-4">
<summary class="cursor-pointer font-semibold">Clip 0130</summary>
<table class="w-full my-2">
<tr><td class="py-1">alpha = -15 (toward major)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0130/alpha_-15.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = -02</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0130/alpha_-02.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +00 (baseline)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0130/alpha_+00.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +02</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0130/alpha_+02.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +15 (toward minor)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0130/alpha_+15.wav" type="audio/wav"></audio></td></tr>
</table>
</details>

<details class="my-4">
<summary class="cursor-pointer font-semibold">Clip 0925</summary>
<table class="w-full my-2">
<tr><td class="py-1">alpha = -15 (toward major)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0925/alpha_-15.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = -02</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0925/alpha_-02.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +00 (baseline)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0925/alpha_+00.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +02</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0925/alpha_+02.wav" type="audio/wav"></audio></td></tr>
<tr><td class="py-1">alpha = +15 (toward minor)</td><td><audio controls class="w-full max-w-sm"><source src="/steering_experiments/clip_0925/alpha_+15.wav" type="audio/wav"></audio></td></tr>
</table>
</details>

# Discussion

## Cross-Architecture Consistency Suggests General Tonal Encoding

The central finding of this work is that tonal encoding is architecturally invariant. MusicGen and DiffRhythm have fundamentally different generation mechanisms—autoregressive token prediction versus parallel latent denoising—and different training objectives. Yet both exhibit the same interpretability signature: mode is linearly decodable from intermediate layers, probe accuracy follows a middle-layer bump peaking at 44–50% network depth, and late layers show declining tonal accessibility. This convergence across architectures suggests that distinguishing major from minor is useful for predicting musical continuations regardless of generation paradigm. The models appear to learn functionally similar representations because tonal structure is genuinely informative for the task, not because of shared architectural biases.

This finding has practical implications. Interpretability tools and intuitions developed for one music generation architecture may transfer to others, reducing the cost of analyzing new models. The similar relative depths at which tonal information peaks (44% for MusicGen, 50% for DiffRhythm) suggest that "middle layers host abstract musical concepts" may be a general principle applicable across the field.

## Why Accuracy Is Modest but Meaningful

Peak probe accuracies of 62% and 59% may seem modest compared to classification tasks where models achieve >90% accuracy. However, these numbers must be interpreted in context. Our "ground truth" labels come from `librosa`'s automatic key detection, which itself achieves only ~70% accuracy on clean recordings and likely performs worse on short synthetic clips with potential generation artifacts. The probe cannot outperform its supervision. Additionally, some clips are genuinely ambiguous between relative major and minor keys—A minor and C major share exactly the same notes and differ only in which note receives emphasis. Finally, tonal information may be partially distributed across many dimensions or encoded in partially nonlinear ways that a simple linear classifier cannot fully recover.

The extreme statistical significance ($p < 10^{-150}$) is the more important metric than absolute accuracy. With 1,024 bootstrap samples and tight confidence intervals, we can definitively reject the null hypothesis that these representations contain no tonal information. The modest accuracy reflects ceiling effects from label noise and inherent ambiguity, not absence of signal.

## Hierarchical Processing in Music Models

The layer-wise pattern we observe—low accuracy in early layers, peak in middle layers, decline in late layers—mirrors findings from NLP interpretability (Tenney et al., 2019)[https://arxiv.org/abs/1905.05950]. This suggests music models exhibit similar hierarchical processing: early layers focus on local features like token embeddings and spectral patterns; middle layers integrate temporal context to form abstract musical concepts including key, mode, and style; late layers transform these abstractions into the specific format needed for output (next-token logits or denoising updates). For controllable generation, this identifies middle layers as the most promising intervention targets—modifying representations at these depths affects abstract musical properties before they are "baked into" low-level output decisions.

## Steering Reveals Causality and Entanglement

The 23% mode flip rate confirms that probe-identified representations are not merely epiphenomenal—they causally influence model outputs. The failure of random directions and non-optimal layers to produce similar effects rules out generic perturbation artifacts. However, informal listening revealed that steering also affects timbre, dynamics, and harmonic complexity alongside mode. The "mode direction" derived from our probe appears to bundle multiple correlated musical attributes that co-occur in training data: minor keys in Western music are associated with darker timbres, slower tempos, and more complex harmonies. Single-direction steering thus provides coarse "mood" control rather than surgical tonal manipulation. Achieving fine-grained independent control over multiple musical attributes will require more sophisticated approaches—perhaps multi-task probing to identify orthogonal concept directions, or concept bottleneck architectures that explicitly disentangle musical properties.

## Implications for Controllable Music Generation

Our findings suggest that lightweight, training-free control over tonal properties is feasible. Rather than fine-tuning models on curated datasets or engineering complex conditioning mechanisms, simple activation steering may provide a path to coarse stylistic control with minimal computational overhead. For creative applications, this could enable rapid prototyping of controllable generation interfaces—a "mood slider" that adjusts the brightness or darkness of generated music without retraining.

The cross-architecture consistency has practical implications for tool development. If tonal representations emerge similarly across MusicGen and DiffRhythm, interpretability techniques may generalize across the diverse ecosystem of music generation models. Analysis pipelines developed for one model could transfer to new releases, reducing the effort required to understand and control each new system.

More broadly, our results demonstrate that music generation models learn structured representations that support inspection and intervention. These are not simply black boxes that happen to produce pleasant audio—they contain organized internal states that encode musically meaningful concepts and can be read out and modified. As generative music tools become more prevalent in production workflows, this understanding becomes crucial for building systems that are not only capable but also controllable and predictable.

# Limitations

Several limitations qualify our conclusions and suggest directions for future investigation.

The most significant limitation concerns label quality. All key and mode labels derive from `librosa`'s automatic detection algorithm, which achieves only ~70% accuracy on clean recordings and likely performs worse on short synthetic clips. The algorithm exhibits systematic biases toward common keys (C major, A minor) and struggles with the ambiguity between relative major/minor pairs. This label noise creates a ceiling on probe accuracy and introduces systematic errors that may bias our layer-wise comparisons. The true amount of tonal information encoded in these models is almost certainly higher than what we can measure with noisy supervision.

The dataset itself limits generalization. We focus exclusively on 5-second solo-piano clips generated from classical/romantic prompts, a deliberately homogeneous setting that simplifies key detection but restricts our findings to this narrow musical domain. Whether the same interpretability patterns hold for longer compositions, multi-instrument arrangements, non-Western tonalities, or highly chromatic and atonal music remains untested. The SAE experiments suffer particularly from this limitation—the 344 usable clips provide grossly insufficient diversity for an 8,192-feature dictionary.

Our temporal pooling strategy discards fine-grained structure. By averaging or max-pooling across time, we lose information about how tonal encoding evolves within a clip—whether key is established in early tokens and maintained, or whether it emerges gradually. Future work could probe per-token or per-frame representations to examine temporal dynamics of tonal encoding.

Steering evaluation relies on automatic key detection rather than human listening studies. The 23% flip rate measures changes in `librosa`'s output, but human perception of mode shift may differ substantially from algorithmic detection. Formal listening studies with musically trained participants would provide stronger evidence for the perceptual impact of steering.

Finally, the steering direction is entangled with correlated musical attributes. We cannot claim to have isolated mode from timbre, dynamics, or harmonic complexity. This entanglement is arguably intrinsic to how music works—minor keys in Western music genuinely co-occur with darker sounds—but it limits the precision of our control mechanism.

# Future Work

Several directions follow naturally from our findings and limitations.

Better supervision is the highest-priority improvement. Human-annotated key labels for a subset of clips would establish a cleaner ground truth, enabling sharper probe accuracy estimates and more confident claims about cross-layer differences. Formal listening studies could quantify perceptual impact of steering and reveal whether "measured flip" corresponds to "perceived flip."

Scaling SAE training to larger, more diverse datasets may enable feature discovery that failed here. A dataset of 10,000+ clips spanning multiple instruments, genres, and tonal systems might provide sufficient diversity to populate an overcomplete dictionary. Alternatively, architectures designed for small-data regimes—gated SAEs, β-VAEs, or regularizers encouraging dictionary utilization—may work with current data sizes.

Temporal analysis could reveal how tonal information flows through generation. Rather than pooling across time, probing individual tokens would show when key is established and how it is maintained. Attention pattern analysis might reveal which heads specialize in tonal processing.

Disentangling musical attributes is essential for precise control. Multi-task probing—simultaneously predicting mode, tempo, timbre, and dynamics—could identify directions that isolate each concept. Concept bottleneck approaches that explicitly factorize representations may enable independent manipulation of different musical properties.

Cross-model transfer experiments would test whether steering vectors learned from MusicGen work in DiffRhythm or other models. Successful transfer would suggest shared tonal representations across the field, enabling efficient development of controllable generation tools.

# Conclusion

We investigated whether text-to-music models encode major/minor mode and whether this encoding supports controllable generation. Our experiments provide affirmative answers to both questions, while also revealing important limitations of current interpretability methods in the audio domain.

The probing results establish that mode is linearly decodable from intermediate activations of both MusicGen (62.1% accuracy at layer 21/48) and DiffRhythm (59.3% at block 12/24), with statistical significance exceeding $p < 10^{-150}$ across 1,024 bootstrap samples. The characteristic middle-layer peak occurs at remarkably similar relative depths—44% for MusicGen and 50% for DiffRhythm—despite these models having radically different architectures and training objectives. Autoregressive token prediction and parallel latent denoising converge on similar internal organizations of tonal information, suggesting that distinguishing major from minor is functionally useful for predicting musical continuations regardless of the specific generation mechanism.

The steering experiments confirm that these representations are not merely epiphenomenal correlates but causally influence model outputs. Adding probe-derived direction vectors to layer 22 activations shifts detected mode with 23% flip rate at $\alpha=15$, while random directions produce negligible effects (< 2%) and steering at non-optimal layers produces weak effects (< 5%). This specificity rules out generic perturbation artifacts and establishes that the identified direction genuinely encodes mode-relevant information that the model uses during generation.

The sparse autoencoder analysis yields a valuable negative result. With 98% of dictionary features remaining dead and no surviving features showing tonal specificity, we demonstrate that SAE methods cannot be naively transferred from language to audio without accounting for dramatic differences in data scale and diversity. Language SAEs train on billions of tokens; our 344-clip dataset was insufficient to populate an 8,192-feature dictionary. This finding motivates future methodological development specifically adapted to audio interpretability.

Taken together, these results demonstrate that standard interpretability tools—probes and steering—provide meaningful traction on music generation systems, while highlighting that feature discovery methods require substantial adaptation. The cross-architecture consistency suggests that tonal encoding may be a general property across the field, enabling efficient development of analysis and control techniques applicable to new models as they emerge. As music generation systems become increasingly prevalent in creative workflows, understanding their internal representations becomes essential for building tools that are not only capable but also transparent, controllable, and aligned with human musical intentions.
