## Rano: Restorable Speaker Anonymization via Conditional Invertible Neural Network

Jianzong Wang, Xulong Zhang /a0 , Xiaoyang Qu Ping An Technology (Shenzhen) Co., Ltd., China

Abstract -Speech contains ample information, including the primary semantic content and information about the speaker, such as gender, age and health status. Speaker-dependent information partially carries personal privacy and has raised concerns about the protection of voice privacy. Speaker anonymization aims to conceal the speaker's identity in speech, while preserving speaker-independent information to the greatest extent possible, and has become an increasingly important task in the field of speech. Existing research generally treats speaker anonymization as a downstream task of voice conversion, often employing speech representation disentanglement-based methods to separate speaker-dependent and speaker-independent information. However, speech representation disentanglement, especially for speakerindependent information, faces challenges such as information leakage or excessive disentangling, resulting in quality degradation. In this paper, we propose a speaker anonymization model called Rano, which does not rely on precise disentanglement. Rano employs a generative invertible neural network to forge anonymous speaker identities from keys and then uses the speaker embeddings as conditions to guide the speaker anonymization process via a conditional invertible neural network. Moreover, when the key is provided, lossless restoration from anonymized speech to original speech can be achieved via the reverse network, thereby expanding the application scenarios of Rano. Experiments demonstrate that the proposed model achieves comparable performance to existing state-of-the-art models. We also verify the security guaranteed by the key in the restoration process.

Index Terms -speaker anonymization, voice privacy, speech security, conditional invertible neural network, generative speaker embedding

## I. INTRODUCTION

Speaker anonymization aims to hide the speaker identity of a given speech while maintaining the information unrelated to the speaker such as content unchanged. As the speech contains not only semantic information but also acoustic information that involves ample information about the speaker, e.g. gender, age, geographic region, personality and even health condition [1]. The leakage of certain private information results in implications and threats to speakers. As malicious attackers use the obtained personal information from speech to form social engineering attacks, others collect target victims' speech to get access to speaker verification systems for further attacks [2]. With the increasing emphasis on personal privacy security [3], various aspects of privacy protection have emerged and played significant roles in the cyber world. The protection of speech is also gradually needed by service providers. A typical application scenario of a speaker anonymization system is the uploading of user speech data by smart devices, as service providers need a large amount of real speech data for semantic understanding systems' training. But the security prerequisite of collecting user data is to retain the user's personal information ( i.e. voice privacy) [4].

/a0 Corresponding author: Xulong Zhang (zhangxulong@ieee.org).

An ideal speaker anonymization model only changes the speaker-dependent information of the original speech while ensuring no significant degradation in quality or naturalness. Speaker anonymization model can be considered as the counterpart of the automatic speaker verification (ASV) model that extracts speaker identity representations to distinguish between speakers, since the speaker anonymization model tries to fool the ASV into not being able to distinguish the original speaker by anonymizing the speech.

VoicePrivacy Challenge (VPC) [5]-[7] brings more concrete requirements to speaker anonymization, as well as rigorous and uniform evaluate metrics, allowing this task to gain the attention of researchers and become an attractive new task in the field of speech. Previous speaker anonymization methods can be divided into two categories, (i) signal processing-based methods and (ii) deep learning-based methods. The former [8][10] leverage traditional signal process techniques to modify the characteristics like pitch and spectral envelope of the original speech without training models. These methods face the challenge of unnaturalness and vulnerability [11]. While the latter [12], [13] often disentangle representations of speakerdependent and speaker-independent information and conduct speech synthesis with another speaker ( i.e. anonymous speaker) representation selected from a speaker pool with certain criterion [14], or obtained via generative model [13]. These disentanglement-based methods treat speaker anonymization as a downstream task of voice conversion (VC) [15]. These methods are built on the assumption that the speech can be decoupled into speaker-dependent and speaker-independent representations, and the privacy information is contained in the speaker-dependent representation. So the protection of voice privacy is achieved by replacing the speaker-related representation.

However, speech representation disentanglement has inherent limitations. The extraction of representation is always conducted on the intermediate features after down-sampling. The low-dimension representations are integrated for speech synthesis, but information loss is inevitable during the extraction process. So the disentangle-combine-synthesize pipeline for voice conversion or anonymization suffers speech quality degradation. But the application scenario mentioned above, where the speaker anonymization system will be used for collecting data for training models, this places demands on the authenticity and clarity of anonymized speech.

Fig. 1: The pipeline of SRD-based speaker anonymization models.

<!-- image -->

In this paper, we focus on the quality and restoration of the anonymized speech and propose a R estorable speaker ano nymization model called Rano . The model leverages a conditional invertible neural network (cINN) for the anonymization process without explicit content disentanglement further to alleviate information loss. Thanks to the characteristics of cINN, the anonymized speech can be restored to the original speech with the reverse network. We define the key as the input used to generate the anonymous speaker embedding and serves as a guarantee of the security of the restoration. The contribution of our work is as follows:

- We design a speaker anonymization model, Rano , which is quite different from the current disentanglementbased methods, with an audio-to-audio transformation to alleviate information loss caused by explicit speech representation disentanglement, further to guarantee the quality of anonymized speech.
- The proposed model is able to restore the original speech from the anonymized speech via the conditional invertible neural network. The ability of restoration enriches application scenarios of our speaker anonymization model.
- We introduce the concept of key for generating speaker embedding as anonymization condition. And the security of restoration complies with Kerckhoff's principle.

## II. RELATED WORK

## A. Speaker Anonymization

The speaker anonymization task is proposed to confuse a malicious attacker so that the attacker cannot obtain real speaker-related information from the anonymized speech. In other words, the speaker anonymization model changes speakerdependent information of the original speech but keeps other information like content unchanged. Speaker anonymization tasks can be divided into speaker-level anonymization and utterance-level anonymization. The former anonymizes all utterances of a speaker to the same anonymous speaker whereas the latter cannot guarantee that the utterances of the same speaker are identity-consistent after anonymization. Signal processing is one of the solutions to speaker anonymization. Signal processing-based methods [8]-[10] try to change the speaker identity information through modifications on speech characteristics like pitch and envelope. Gupta et al. [8] use McAdam's coefficient to tune the pole angle and radius of the linear prediction model of speech to achieve anonymization. Patino et al. [9] introduce an efficient method starting with the modification of the spectral envelope of the speech signal.

Another series of anonymization methods is based on deep learning. Most of these works treat speaker anonymization as the downstream task of voice conversion, the VC models process utterances sound as if they are spoken by other speakers [16], according to the target speech. Speech representation disentanglement (SRD) [17], [18] shows its flexibility in VC tasks, and is widely leveraged for various VC models. Consequently, current speaker anonymization models follow in the footsteps of VC, several SRD-based speaker anonymization models are introduced. The pipeline of SRD-based speaker anonymization is shown in Fig. 1, representation of speakerindependent information ( i.e. content) is extracted with an encoder and combined with pseudo-speaker representation that is selected from a speaker pool according to specific criteria such as the largest distance, or generated via a generator. Then the representation is sent to the decoder to integrate and synthesize anonymized speech. Srivastava et al. [14] introduce an x-vector [19]-based model and discuss how to select x-vectors from the speaker pool to be used as anonymous speaker representations. They explore different distance criteria as well as selection strategies and strike a balance between anonymity effect and practicality. Meyer et al. [20] utilize an automatic speech recognition (ASR) model to extract phonetic transcriptions as speaker-unrelated representation to synthesize anonymized speech. Mavalim et al. [10] modify the fundamental frequency to reduce the residual speaker-dependent information after disentanglement. Meyer et al. [21] obtain pseudo-speaker embeddings from a Generative Adversarial Network (GAN) for anonymization. Yuan et al. propose DeIDVC [13], which builds autoencoders to decouple content and speaker embeddings, a generator trained on speaker embeddings replaces the speaker encoder during the anonymization process. However, accurately disentangling various information from speech is challenging, SRD-based methods often suffer from insufficient or excessive disentangling, which leads to serious performance degradation.

## B. Invertible Neural Network

The prototype of Invertible Neural Network (INN) is formed in NICE [22] and RealNVP [23]. The coupling layer, as well as the tractable Jacobian determinants, is of importance, enabling maximum likelihood training. The invertible block divides the input into two parts, then conducts non-linear transformations as well as addition and multiplication operations between them to achieve transforming the space. Kingma i.e. propose GLOW [24] that leverages invertible 1 × 1 convolution to enhance the generative model's tractability of log-likelihood, further improving the validity of the synthesized images. Ardizzone et al. [25] prove that INN can be leveraged to estimate the full posterior of an inverse problem, thus bringing the application of INN to scale. Later, they [26] add the condition to the internal function of the coupling layer to build Conditional Invertible Neural Network (cINN) for image generation.

INN can be widely used in generation tasks [22]-[24], [27], where INN fits an invertible y = f θ ( x ) to build transformation between x and z . Invertible means that p X ( x ) can be sampled through the inverse process of f θ as x = f -1 θ ( y ) , where f θ and f -1 θ share parameters θ . Since we can predefine the density function and the training process just maximizes the likelihood.

In addition, INN shows its ability in other image-to-image or audio-to-audio processing tasks. Lugmayr et al. [28] propose an invertible flow-based model for super-resolution. Xiao et al. [29] make use of the information in down-scaling for the same image's up-scaling with an invertible re-scaling model. INN is also leveraged for image watermarking [30] and audio watermarking [31]. Guan et al. [32] concatenate several INN for image steganography, the extraction process follows the reverse order of the hiding process. As one of the speech-tospeech transformation tasks, the speaker anonymization model outputs anonymized speech according to the given speech, in this paper, we utilize cINN to complete anonymization, rather than an SRD-based encoder-decoder structure that requires explicit and accurate disentanglement.

## III. METHOD

## A. Overall Architecture of Rano

Fig. 2: The architecture of Rano. The Restorer and Anonymizer have opposite network structures and share parameters. ACG and ASV are pre-trained. ACG generates speaker embeddings as anonymization conditions from keys for the Anonymizer. A speaker encoder is utilized as ASV to extract speaker embeddings for contrastive learning. The training process does not require a vocoder, which is omitted in the figure.

<!-- image -->

The architecture of the proposed speaker anonymization model, Rano, is showcased in Fig. 2. Different from the SRDbased model, Rano achieves anonymization in a speech-tospeech transformation manner, which performs no speakerindependent representation extraction nor representation integration. The input speech is firstly transformed to the time-spectral domain as Mel-spectrogram via Short-Time Fourier Transformation (STFT). The anonymizer which is designed in an invertible structure processes the input with the anonymization conditions provided by the Anonymization Condition Generator (ACG). ACG generates speaker embedding with a key to put restrictions on the generation so that the correct restoration process cannot be achieved without the correct key. The generated speaker embedding serves as the condition and guides the anonymizer to perform the speaker-level or utterance-level speaker anonymization process. The anonymized speech and the condition are then sent to the restorer for restoration. The restorer owns the opposite network structure to the anonymizer and they share the same parameters, so the restored speech can be exactly the same as the original speech if the same condition is input. Contrastive learning is conducted on the anonymized speech and the original speech, where the speaker embedding extracted from the anonymized speech serves as the anchor, the embedding from the original speech and the condition play the roles of negative sample and positive sample, respectively. Speaker embeddings are extracted from the Mel-spectrogram speech via a lightweight pre-trained speaker encoder ( i.e. ASV), the contrastive learning enlarges the difference between the speaker identity of the original speech and anonymized speech and guides the neural network to modify the speaker information of the original speech according to the condition.

## B. cINN-based Identity Transformation

Rano is built on a Conditional Invertible Neural Network (cINN) structure. ACG generates anonymization condition with a key for the anonymizer, facilitating the network to transform the speaker identity of the original speech to an anonymous speaker identity. The anonymizer and ACG both consist of several INN blocks or cINN blocks as shown in Fig. 2. Conditions are inserted to the outputs of the sub-nets ( i.e. the network modules ψ, ϕ, ρ, η in Fig. 2) inside coupling layer. The sub-nets are not invertible and execute the same operations in the forward and backward processes, so the same condition needs to be input in both of forward and backward processes.

The forward process of each cINN block of the anonymizer is as follows,

<!-- formula-not-decoded -->

where u and v are split from the input Mel-spectrogram in the channel dimension, u i +1 and v i +1 are obtained from the i -th block with u i and v i as inputs, ψ ( · ) , ϕ ( · ) , ρ ( · ) and η ( · ) are internal functions implemented by neural network module like RRDB [33] and make no effects on the invertibility of the cINN block, and c denotes the anonymization condition. Consequently, the backward process ( i.e. the process of the restorer) performs the opposite operation as shown in Eq. 2.

<!-- formula-not-decoded -->

For simplicity, we denote the anonymizer as f ( · ; c, θ ) with the condition c and trainable parameters θ , the restorer which owns the reverse structure and the same parameters to the anonymizer is denoted as f -1 ( · ; c, θ ) . Thus, the anonymization and restoration processes can be represented as x a = f ( x ; c, θ ) and x r = f -1 ( x a ; c, θ ) , where x , x a and x r denote the original speech, the anonymized speech and the restored speech, respectively. Thanks to the parameter-sharing characteristic of INN, the restored speech could be the same as the original speech, if the condition is consistent with the anonymization process and the anonymized speech remains unchanged.

## C. The Key and Anonymization Condition

According to Kerckhoffs's principle , the security of a cryptographic algorithm is guaranteed by the key rather than the secrecy of the algorithm or system. The same to the security of restoration, we introduce the key to impose restrictions on Rano's restoring ability. As mentioned above, the condition c guides the anonymization, and in the restoration process, the same condition must be fed to the reverse network to perform the correct backward process. For Rano, c is generated by the Anonymization Condition Generator (ACG), and we design the key to determine c , thus whether the restoration process can be performed correctly depends on whether the correct key is present.

ACG is an INN-based generative model that is able to perform invertible transformation between the latent space and the condition space ( i.e. speaker embedding space). We denote the network of ACG as f acg ( · ; ω ) with trainable parameters ω . As f acg ( · ; ω ) is invertible, during the training stage, f acg ( · ; ω ) builds transformation from speaker embedding s that follows certain distribution to a latent variable that follows a prescribed simple distribution. Each input s is assigned with a probability by f acg ( · ; ω ) , as shown in Eq. 3, where det ( ∂f acg ∂s ) is the determinant of the Jacobian matrix.

<!-- formula-not-decoded -->

The training of ACG aims to maximize the logarithm of the posterior p ( ω ; s ) [26]. According to Bayes' theorem, we minimize the loss in Eq. 4, where J i is the determinant of the Jacobian matrix when given sample s i , as J i := det ( ∂f acg ∂s i ) , the p Z ( z ) is set to the standard normal distribution, and τ = 1 2 σ 2 ω .

<!-- formula-not-decoded -->

After the training is completed, ACG utilizes the reverse process f -1 acg ( · ; ω ) that shares the same parameters ω with f acg ( · ; ω ) to generate anonymous speaker embedding s a that follows the distribution of the real speaker embeddings with the key that is sampled from the standard normal distribution as input, as s a = f -1 acg ( key ; ω ) , key ∼ N (0 , 1) .

The key determines the outputs of ACG and enables Rano to perform speaker-level or utterance-level anonymization tasks. When the same key is used for the utterances from one speaker, ACG generates the same condition for the utterances so that the anonymized speech appears to be spoken by the same anonymous speaker, whereas when the key for each utterance is different from another, each anonymized speech owns different speaker identity.

## D. SII-Consistency and SDI-Differentiation

Essentially, the speaker anonymization model centres on keeping the speaker-independent information (SII) of speech unaltered while changing the speaker-dependent information (SDI). Rano achieves speaker anonymization without disentangling SII representations considering the information loss or redundancy caused by disentanglement.

To ensure that SII remains unchanged after processing without using parallel speech ( i.e. the same utterances from different speakers) datasets or alignment for training, we leverage consistency loss for the training of Rano. After processing the original speech with the condition generated via ACG, the original speech is fed to the anonymizer again but with the speaker embedding extracted from the original speech as the condition. In this case, the anonymizer is expected to make as few changes as possible to the original speech. In this manner, cINN learns to only transform the SDI according to the condition but maintain SII unchanged without disentangling that separately. The consistency loss which guarantees the SII-consistency is shown in Eq. 5,

<!-- formula-not-decoded -->

where s denotes the real speaker embedding that is extracted from the original speech and is fed to the anonymizer as the condition.

To guide Rano to gain the ability to convert SDI of the speech to the SDI involved in the condition, contrastive learning [34] is leveraged to enhance the distinction of SDI between the original speech and the anonymized speech. As one of the experiences of contrastive learning, chosen positive and negative samples are expected to be confusing ( i.e. the 'hard examples'). For the training of Rano, firstly, we select SDI extracted from the anonymized speech as the anchor. The generated speaker embeddings and the original speaker embeddings are set as positive samples and negative samples, respectively. The triplet loss [35] is shown in Eq. 6.

<!-- formula-not-decoded -->

where f asv ( · ) denotes the ASV model which extracts speaker embeddings from Mel-spectrograms differentially, d ( · ) is a distance criterion that measures the distinction between two variables, c denotes the anonymization condition or generated speaker embedding, and κ is the threshold. The goal is to enlarge the distance between the original SDI and anonymized SDI, while reducing the difference between anonymized SDI and the expected SDI of the anonymous condition.

## E. Training Strategy

Fig. 3(a) and Fig. 2 showcase the two-stage training strategy for Rano. As for ACG, an INN-based generative model, requires pre-training to gain the ability to generate data ( i.e. speaker embeddings as conditions for the anonymizer) from data that follows prescribed distribution. The loss function for the first stage is shown in Eq. 3.

Fig. 3: The training strategy and inference stage of Rano. (a) ACG is pre-trained with an ASV that extracts speaker embeddings. (b) The inference stage of Rano, involves the anonymization and restoration processes.

<!-- image -->

## Algorithm 1 Loss acquisition in the 2 nd training stage.

Input

: original speech x

Module : pre-trained f asv , f acg , under-training

f ano , f res (:= f - 1 ano )

Parameter : thresholds for ACG d , for contrastive learning κ , loss weights λ 1 , λ 2

Output

: total loss L total

- 1: Initialize the variables: Initialize Key : Key ← Random ( N (0 , 1)) Initialize c : c ← f acg ( Key ) Extract original speaker embedding s : s ← f asv ( x )
- 2: while | s -c | &lt; d do
- 3: Key ← Random ( N (0 , 1)) , c ← f acg ( Key )
- 4: end while
- 5: Forward process 1: x ano ← f ano ( x , c )
- 6: Forward process 2: ˆ x ← f ano ( x , s )
- 7: Obtain losses:

Get consistency loss L cons with { x , ˆ x } as in Eq. 5 Get triplet loss L tri with { κ , f asv ( x ano ) , c , s } as in Eq. 6

- 8: L total ← λ 1 L cons + λ 2 L tri
- 9: return L total

In the second training stage, the pre-trained ACG is fixed and leveraged for generating conditions for the anonymizer as well as the restorer with a randomly sampled key from the set distribution ( i.e. standard normal distribution). Then the anonymizer processes the original speech according to the generated condition. Note that the restoration ability of the restorer is derived from the anonymizer, without extra training. The pseudocode of the workflow of obtaining each loss in the second training stage is shown in Alg. 1. The pre-trained ASV in two stages should be the same. Combining the triplet loss introduced in Eq. 6 and the consistency loss in Eq. 5 with weights λ 1 and λ 2 , the total loss for the second training stage is shown in Eq. 7,

<!-- formula-not-decoded -->

## A. Settings

- 1) Data: We conduct Rano's training on VCTK corpus [36] and LibriTTS dataset [37]. The former contains over 100 hours of speech data from over 100 different speakers, while the latter comprises over 500 hours of speech data taken from audiobooks. About 15% of the data in each dataset is separated as the test set. All the utterances are resampled to 22,050 Hz during our experiments.
- 2) Training: Two Adam optimizers [38] with β 1 = 0 . 9 , β 2 = 0 . 99 , ϵ = 10 -8 , lr = 10 -5 are leveraged for the two stages of training. A StepLR scheduler is also leveraged in the second training stage to adjust the learning rate as training progresses. The weights λ 1 , λ 2 for the total loss in Eq. 7 are set to 1 and 5 respectively. The training for ACG contains 100,000 iterations while the second training stage contains 200,000 iterations. A pre-trained lightweight ASV from the speaker encoder of AdaIN-VC [15] is used to obtain speaker embeddings from Mel-spectrograms, providing training data for ACG in the first training stage as in Fig. 3(a) as well as extracting original speaker embeddings for Rano in the second stage to build consistence loss and conduct contrastive learning. We utilize Hifi-GAN [39] as the vocoder to convert Mel-spectrograms into waveforms to obtain anonymized speech and restored speech.
- 3) Comparison Methods: We compare Rano with several deep learning-based models. The baseline model of VPC 2022 [7] (B1.a) is one of the comparison models. DeID [13] and SALT [11] are the latest speaker anonymization models and are chosen to compare with Rano. All the models are trained with the same settings in Section IV-A. Consistent with VPC 2022 [7] requirements, we evaluate the performance of the models on the test sets , which do not appear in the training data.

## B. Privacy Evaluation

- 1) Objective evaluation: As the adversary of speaker anonymization, speaker verification model tries to discriminate the speaker identity of the anonymized speech, and an equal error rate ( EER ) is obtained when the flase acceptance rate equals to false rejection rate . Higher EER indicates anonymized speech poses a greater challenge to the speaker verification model, i.e. better anonymization performance. A pre-trained ASV from SpeechBrain [40] is employed to perform speaker verification by directly extracting ECAPA-TDNN embeddings from waveform. It's worth noting that although training Rano requires the participation of a pre-trained ASV, the ASVs for training and evaluation are completely different for the fairness of comparison.

For the speaker anonymization model, the anonymous speakers corresponding to different original speakers should exhibit diversity. Gain of voice distinctiveness ( G V D ) measures the speaker distinctiveness before and after the anonymization process. It is obtained from two matrices that respectively contain speaker distinctiveness in original and anonymized

## IV. EXPERIMENT

TABLE I: Comparison of different speaker anonymization methods and comparison results for ablation study.

| Method            |   EER (%) ↑ |   WER (%) ↓ | G VD ( dB )   | ρ f 0 ↑   |   naturalness ↑ |   intelligibility ↑ | restorable   |
|-------------------|-------------|-------------|---------------|-----------|-----------------|---------------------|--------------|
| Ground-Truth      |        3.20 |        6.58 | -             | -         |            4.45 |                4.32 | -            |
| VPC B1.a [7]      |       17.40 |       16.21 | -10.34        | 0.73      |            3.13 |                3.27 |              |
| DeID [13]         |       41.46 |       13.04 | 0.21          | 0.78      |            3.35 |                3.40 |              |
| SALT [11]         |       45.32 |       10.77 | -0.03         | 0.83      |            3.71 |                3.87 |              |
| Rano ( proposed ) |       47.81 |       11.91 | 0.39          | 0.80      |            3.73 |                3.84 |              |
| Rano w/o ACG      |       32.04 |       11.97 | -4.33         | 0.77      |            3.09 |                3.73 |              |

spaces [41]. When G V D values 0 dB , the distinctiveness between original speakers is preserved in anonymous speakers. Increments in distinctiveness result in gains above 0 , while degradation results in gains below 0 . An ideal model owns G V D values close to 0 or above.

To evaluate the retention of speaker-independent information, we measure the word error rate ( WER ) and pitch correlation ( ρ f 0 ). WER with a smaller value indicates more complete content retention. We employ Whisper [42] (large) speech recognition model to measure WER in our experiment. Pitch correlation measures the consistency of the fundamental frequency before and after the anonymization process. It is calculated via the Pearson correlation between two pitch sequences. A higher pitch correlation indicates better preservation of the pitch information.

2) Subjective evaluation: Objective evaluation metrics can not always reflect the true performance of speech models because the human auditory system is complex and sensitive. Subjective evaluation can compensate for the limitations of objective evaluation and is equally important as objective evaluation. Mean opinion score (MOS) is widely used in the speech field. For speaker anonymiztion, naturalness ( MOS -N ) , intelligibility ( MOS -I ) and verifiability ( MOS -V ) are human-evaluated [7]. Naturalness measures how natural the speech sounds, considering any artefacts or degradation introduced by processing tools. Intelligibility measures how clearly and easily the content of the speech can be understood. V erifiability measures how similar the speaker identity in two audio samples are, determining whether they sound from the same speaker or different speakers. Eleven participants are invited to evaluate the naturalness and intelligibility of anonymized speech and rate them with scores from 1 to 5 . High scores for naturalness or intelligibility indicate higher speech quality.

## C. Results

The measurement of Rano is conducted in speaker-level , that is, we provide each speaker with a key for all of his/her utterances to ensure each original speaker is related to only one anonymous speaker during the measurement. The comparisons of different speaker anonymization models are shown in Tab. I. The results show that Rano obtains comparable anonymization quality to the state-of-the-art methods and shows ideal anonymous speaker distinctiveness and speech quality and also preserves speaker-independent information like content and pitch information in anonymization processing.

Tab. II showcases the verification of Rano in several cases. When the two utterances in a pair are from the same speaker, Rano anonymizes one of the utterances to make it sound different from the speaker's. When the two utterances from different speakers are anonymized with different keys, they still sound like from different anonymous speakers.

Fig. 4: The Mel-spectrograms of original, anonymized, and restored speech in three anonymization tasks.

<!-- image -->

The Mel-spectrogram samples of three speaker anonymization tasks completed via Rano are shown in Fig. 4. The details of the Mel-spectrograms show that the anonymized speech owns similar linguistic characteristics but different acoustic characteristics which can be observed from the shape and distribution of harmonic. Thanks to the characteristics of cINN, when the same condition is provided in the backward process, we can obtain the Mel-spectrograms almost the same to the original. As shown in Fig. 4, the subtle differences between them result from the effects of the vocoder.

Rano performs speaker-level anonymization when assigning each speaker with a fixed key respectively, so that the anonymous speaker corresponding to the original speaker is fixed. The visualization of speaker distinctiveness of Rano's performance is shown in Fig. 5. The speaker embeddings are extracted while the ASV model from SpeechBrain [40]. We can find that the identity that each speaker transforms to after anonymization is still recognizable from each other.

Fig. 5: Speaker embedding clustering. The same colours in (a) and (b) denote original and correlated anonymous speakers.

<!-- image -->

TABLE II: Verifiability scores of Rano. Scores range from 1 to 5, with higher scores representing a more similar auditory perception of speaker identity.

| speaker    | anonymized   | anonymized   | verifiability   |
|------------|--------------|--------------|-----------------|
| speaker    |              | speech b     | verifiability   |
| same       |              |              | 4.36 ( GT )     |
| different* |              |              | 1.62 ( GT )     |
| same       |              |              | 3.76            |
| different* |              |              | 1.91            |
| same       |              |              | 2.16            |

* Different speaker of the same gender.

## D. Ablation Study and Security Discussion

To validate the effectiveness and security of ACG with its key, we conduct an ablation study and security discussion. Firstly, the following models are built and trained with the same settings:

- Rano w/o ACG : remove ACG from Rano and feed cINN with same-dimension random variables as conditions during training and inference stages.
- Rano : the proposed speaker anonymization model.

The results of the ablation study are shown in Tab. I. From the results, we can find that using random variables as condition leads to a significant degradation in the quality of anonymized speech and unsatisfactory anonymization results. Different from previous speaker anonymization methods, Rano can restore the anonymized speech to the origin with the secret key. This characteristic enlarges the model's application scenarios but may also raise concerns for the security of the model. So we evaluate the security of its restorability.

As mentioned above, Rano restores the anonymized speech under the same conditions as the anonymization process. The guarantee of the condition's secrecy comes from the preservation of the key . To validate the model's security, we simulate attacks where the attacker tries to restore the speech anonymized via Rano to the original speech without correct key. The anonymized speech is fed into the restorer with randomly sampled variables from Gaussian noise as the fake key for generating anonymization conditions. We use cosine distance as the criterion to measure the similarity between the original key and the fake key. The cosine similarity Sim spk between speaker embeddings (extracted via the pre-trained ASV from SpeechBrain [40]), and Mel-cepstral distortion (MCD) which measures the distance between Mel frequency cepstral coefficient (MFCC), are served as the evaluation metrics. The comparison between the illegally restored speech ( i.e. restored with the fake key) and original speech with different D key interval is shown in Tab. III. Metrics for each case are measured with 100 samples.

TABLE III: Experimental results of security discussion. The measurement is conducted on the waveform generated by the vocoder.

| Case                  |   Sim spk (%) ↑ |   MCD ( dB ) ↓ |
|-----------------------|-----------------|----------------|
| 0 . 4 > D key > 0 . 3 |            4.66 |           7.51 |
| 0 . 3 > D key > 0 . 2 |            9.49 |           5.86 |
| 0 . 2 > D key > 0 . 1 |           12.78 |           5.17 |
| Ground-Truth*         |           87.75 |           1.97 |

Note that when D key = 0 ( i.e. Ground-Truth ), the restored Mel-spectrogram before feeding to the vocoder is exactly the same as the original Mel-spectrogram which is obtained from the original speech, the subtle difference between their speaker embeddings and the non-zero MCD are resulted from vocoder as the speaker embeddings are extracted from the waveforms produced by vocoder, and MCD is also computed from the final waveforms. Results show that the illegally restored speech is significantly different from the normally restored speech in the speaker identity and MFCC, which indicates that the consistent key to the anonymization process is necessary for the accurate restoration. So the security of the model's restorability depends on the secrecy of the key rather than the secrecy of the anonymization method or trained model.

## V. CONCLUSION

In this paper, we propose a speaker anonymization model named Rano. The conditional invertible neural network-based model performs anonymization and restores the original speech from the anonymized speech through its backward process. The security of access to correct restoration depends on the secrecy of the key which is input to the ACG for generating anonymization condition. Experimental results show that Rano achieves satisfactory anonymization quality from both objective and subjective metrics. Rano obtains not only ideal speaker anonymization performance but also the ability to restore original speech from anonymized speech without information loss, which enlarges the application scenarios of the speaker anonymization model.

## VI. ACKNOWLEDGMENT

This work was supported by the National Key Research and Development Program of China (Youth Scientist Project) under Grant No. 2024YFB4504300 and the Shenzhen-Hong Kong Joint Funding Project (Category A) under Grant No. SGDX20240115103359001. Corresponding author is Xulong Zhang (zhangxulong@ieee.org).

## REFERENCES

- [1] R. Khamsehashari, Y. Sinha, J. Hintz, S. Ghosh, T. Polzehl, C. Franzreb, S. Stober, and I. Siegert, 'Voice privacy-leveraging multi-scale blocks with ecapa-tdnn se-res2next extension for speaker anonymization,' in 2nd Symposium on Security and Privacy in Speech Communication , 2022, pp. 43-48.
- [2] Z. Wu, J. Yamagishi, T. Kinnunen, C. Hanilc ¸i, M. Sahidullah, A. Sizov, N. Evans, M. Todisco, and H. Delgado, 'Asvspoof: the automatic speaker verification spoofing and countermeasures challenge,' IEEE Journal of Selected Topics in Signal Processing , vol. 11, no. 4, pp. 588-604, 2017.
- [3] P. Regulation, 'Regulation (eu) 2016/679 of the european parliament and of the council,' Regulation (eu) , vol. 679, 2016.
- [4] P. Champion, A. Larcher, and D. Jouvet, 'Are disentangled representations all you need to build speaker anonymization systems?' in 23rd Annual Conference of the International Speech Communication Association . ISCA, 2022, pp. 2793-2797.
- [5] N. A. Tomashenko, B. M. L. Srivastava, X. Wang, E. Vincent, A. Nautsch, J. Yamagishi, N. W. D. Evans, J. Patino, J. Bonastre, P. No´ e, and M. Todisco, 'Introducing the voiceprivacy initiative,' in 21st Annual Conference of the International Speech Communication Association . ISCA, 2020, pp. 1693-1697.
- [6] N. Tomashenko, X. Wang, E. Vincent, J. Patino, B. M. L. Srivastava, P.-G. No´ e, A. Nautsch, N. Evans, J. Yamagishi, B. O'Brien et al. , 'The voiceprivacy 2020 challenge: Results and findings,' Computer Speech &amp; Language , vol. 74, p. 101362, 2022.
- [7] N. Tomashenko, X. Wang, X. Miao, H. Nourtel, P. Champion, M. Todisco, E. Vincent, N. Evans, J. Yamagishi, and J.-F. Bonastre, 'The voiceprivacy 2022 challenge evaluation plan,' arXiv:2203.12468 , 2022.
- [8] P. Gupta, G. P. Prajapati, S. Singh, M. R. Kamble, and H. A. Patil, 'Design of voice privacy system using linear prediction,' in Asia-Pacific Signal and Information Processing Association Annual Summit and Conference , 2020, pp. 543-549.
- [9] J. Patino, N. A. Tomashenko, M. Todisco, A. Nautsch, and N. W. D. Evans, 'Speaker anonymisation using the mcadams coefficient,' in 22nd Annual Conference of the International Speech Communication Association , 2021, pp. 1099-1103.
- [10] C. O. Mawalim, S. Okada, and M. Unoki, 'Speaker anonymization by pitch shifting based on time-scale modification,' in 2nd Symposium on Security and Privacy in Speech Communication , 2022.
- [11] Y. Lv, J. Yao, P. Chen, H. Zhou, H. Lu, and L. Xie, 'Salt: Distinguishable speaker anonymization through latent space transformation,' in IEEE Automatic Speech Recognition and Understanding Workshop , 2023, pp. 1-8.
- [12] G. P. Prajapati, D. K. Singh, P. P. Amin, and H. A. Patil, 'Voice privacy using cyclegan and time-scale modification,' Computer Speech &amp; Language , vol. 74, p. 101353, 2022.
- [13] R. Yuan, Y. Wu, J. Li, and J. Kim, 'Deid-vc: Speaker de-identification via zero-shot pseudo voice conversion,' in 23rd Annual Conference of the International Speech Communication Association , 2022, pp. 2593-2597.
- [14] B. M. L. Srivastava, N. Tomashenko, X. Wang, E. Vincent, J. Yamagishi, M. Maouche, A. Bellet, and M. Tommasi, 'Design choices for xvector based speaker anonymization,' the 21st Annual Conference of the International Speech Communication Association , pp. 1713-1717, 2020.
- [15] J.-c. Chou and H.-Y. Lee, 'One-shot voice conversion by separating speaker and content representations with instance normalization,' the 20th Annual Conference of the International Speech Communication Association , 2019.
- [16] B. Sisman, J. Yamagishi, S. King, and H. Li, 'An overview of voice conversion and its challenges: From statistical modeling to deep learning,' IEEE ACM Transactions on Audio Speech Language Processing , vol. 29, pp. 132-157, 2021.
- [17] K. Qian, Y. Zhang, S. Chang, M. Hasegawa-Johnson, and D. D. Cox, 'Unsupervised speech decomposition via triple information bottleneck,' in 37th International Conference on Machine Learning , vol. 119, 2020, pp. 7836-7846.
- [18] C. H. Chan, K. Qian, Y. Zhang, and M. A. Hasegawa-Johnson, 'Speechsplit2.0: Unsupervised speech disentanglement for voice conversion without tuning autoencoder bottlenecks,' in IEEE International Conference on Acoustics, Speech and Signal Processing , 2022, pp. 63326336.
- [19] D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudanpur, 'X-vectors: Robust dnn embeddings for speaker recognition,' in IEEE
20. International Conference on Acoustics, Speech and Signal Processing , 2018, pp. 5329-5333.
- [20] S. Meyer, F. Lux, P. Denisov, J. Koch, P. Tilli, and N. T. Vu, 'Speaker anonymization with phonetic intermediate representations,' in 23rd Annual Conference of the International Speech Communication Association , 2022, pp. 4925-4929.
- [21] S. Meyer, P. Tilli, P. Denisov, F. Lux, J. Koch, and N. T. Vu, 'Anonymizing speech with generative adversarial networks to preserve speaker privacy,' in IEEE Spoken Language Technology Workshop , 2023, pp. 912-919.
- [22] L. Dinh, D. Krueger, and Y. Bengio, 'NICE: non-linear independent components estimation,' in 3rd International Conference on Learning Representations, Workshop Track Proceedings , 2015.
- [23] L. Dinh, J. Sohl-Dickstein, and S. Bengio, 'Density estimation using real NVP,' in 5th International Conference on Learning Representations , 2017.
- [24] D. P. Kingma and P. Dhariwal, 'Glow: Generative flow with invertible 1x1 convolutions,' Advances in Meural Information Processing Systems , vol. 31, 2018.
- [25] L. Ardizzone, J. Kruse, C. Rother, and U. K¨ othe, 'Analyzing inverse problems with invertible neural networks,' in 7th International Conference on Learning Representations , 2018.
- [26] L. Ardizzone, C. L¨ uth, J. Kruse, C. Rother, and U. K¨ othe, 'Guided image generation with conditional invertible neural networks,' arXiv:1907.02392 , 2019.
- [27] T. F. van der Ouderaa and D. E. Worrall, 'Reversible gans for memoryefficient image-to-image translation,' in IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2019, pp. 4720-4728.
- [28] A. Lugmayr, M. Danelljan, L. Van Gool, and R. Timofte, 'Srflow: Learning the super-resolution space with normalizing flow,' in Computer Vision-ECCV: 16th European Conference , 2020, pp. 715-732.
- [29] M. Xiao, S. Zheng, C. Liu, Y. Wang, D. He, G. Ke, J. Bian, Z. Lin, and T.-Y. Liu, 'Invertible image rescaling,' in Computer Vision-ECCV: 16th European Conference . Springer, 2020, pp. 126-144.
- [30] R. Ma, M. Guo, Y. Hou, F. Yang, Y. Li, H. Jia, and X. Xie, 'Towards blind watermarking: Combining invertible and non-invertible mechanisms,' in 30th ACM International Conference on Multimedia , 2022, pp. 1532-1542.
- [31] G. Chen, Y. Wu, S. Liu, T. Liu, X. Du, and F. Wei, 'Wavmark: Watermarking for audio generation,' arXiv:2308.12770 , 2023.
- [32] Z. Guan, J. Jing, X. Deng, M. Xu, L. Jiang, Z. Zhang, and Y. Li, 'Deepmih: Deep invertible network for multiple image hiding,' IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 45, no. 1, pp. 372-390, 2022.
- [33] X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, Y. Qiao, and C. C. Loy, 'ESRGAN: enhanced super-resolution generative adversarial networks,' in European Conference on Computer Vision Workshops , vol. 11133, 2018, pp. 63-79.
- [34] A. Jaiswal, A. R. Babu, M. Z. Zadeh, D. Banerjee, and F. Makedon, 'A survey on contrastive self-supervised learning,' Technologies , vol. 9, no. 1, p. 2, 2020.
- [35] F. Schroff, D. Kalenichenko, and J. Philbin, 'Facenet: A unified embedding for face recognition and clustering,' in IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2015, pp. 815-823.
- [36] J. Yamagishi, C. Veaux, and K. MacDonald, 'Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit,' 2016.
- [37] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, 'Libritts: A corpus derived from librispeech for text-to-speech,' in 20th Annual Conference of the International Speech Communication Association , 2019, pp. 1526-1530.
- [38] D. P. Kingma and J. Ba, 'Adam: A method for stochastic optimization,' in 3rd International Conference on Learning Representations , 2015.
- [39] J. Kong, J. Kim, and J. Bae, 'Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis,' Advances in Neural Information Processing Systems , vol. 33, pp. 17 022-17 033, 2020.
- [40] M. Ravanelli, T. Parcollet, P. Plantinga, A. Rouhe, S. Cornell, L. Lugosch, C. Subakan, N. Dawalatabad, A. Heba, J. Zhong et al. , 'Speechbrain: A general-purpose speech toolkit,' arXiv:2106.04624 , 2021.
- [41] P.-G. No´ e, J.-F. Bonastre, D. Matrouf, N. Tomashenko, A. Nautsch, and N. Evans, 'Speech pseudonymisation assessment using voice similarity matrices,' in 21st Annual Conference of the International Speech Communication Association , 2020, pp. 1718-1722.
- [42] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, 'Robust speech recognition via large-scale weak supervision,' in International Conference on Machine Learning , 2023, pp. 28 49228 518.