It looks like there is an issue with the rendering on my markdiwn compiler, please rewrite the script below using \( \) and \[\] for maths in place of $..$ and $$..$$:
# Reviewer zqV1

This paper studies the extension of the Convex Gaussian min-max theorem to non-Gaussian settings. The claim is that this ascertains the scope of Gaussian universality for ERM’s. Gaussian universality essentially means that learning problems under the ERM paradigm can essentially be analyzed as if the data is from a Gaussian distribution with same parameters as the original distribution.
Strengths And Weaknesses:

Strengths

    The work is a contribution to the decade long line of work regarding the Gaussian universality for ERM’s and makes several non-trivial strides in the state of the art knowledge in the area.

Weaknesses -

    The presentation is extremely dense and almost unverifiable.
    A major weakness is that even the problem statement is not clear to a computer science researcher outside of the field despite significant effort. Pointers to surveys, avoiding usage of technical terms without precedence especially in the introduction and "our contributions" section can help with the presentation.
    Nit: In page 2, it should be ‘ERM’ solution instead of ‘EMR’.

Soundness: 3: good
Presentation: 1: poor
Significance: 4: excellent
Originality: 4: excellent
Key Questions For Authors:

    Can you please outline a concrete practical use case and implications of the current work on it?
    I could get the overall gist of the paper and proofs. But could you explain, in simple terms, the content of Theorem 4.3. As stated in the paper, it is a bit dense.
    Assumption 3 seems to be artificial. Can you please say a bit more as to why it makes sense and is natural?


## Rebuttal
Thank you for the positive assessment of the significance and originality. We agree that the current exposition is too dense for a broader ML audience. In the revision, we will simplify the introduction, add a short roadmap with the three notions of universality stated upfront, include broader pointers to CGMT/AMP/RMT, and add a plain-language explanation of Theorem 4.3.

We also understand the concern about verifiability. The anonymous GitHub repository linked in the paper was meant to make the claims easier to check: it reproduces the synthetic and MNIST experiments and includes additional notebooks for several models and losses. We will point readers to it more clearly in the paper as empirical support for the theory.

To make navigation easy, we answer in the order of your questions and use the same labels for answers that also appear in other rebuttals.

1. **Practical use case and simple meaning of Theorem 4.3**

   **Shared response A: practical meaning / Theorem 4.3 / performance universality**

   A concrete use case is high-dimensional classification or regression on structured non-Gaussian features, for example mixture models, random-feature-type maps, or real embeddings such as MNIST features, where Gaussian asymptotics are often used to tune regularization or predict test error. Our result tells us when that shortcut is reliable and when it is not.

   In simple terms, Theorem 4.3 says that the random ERM can be summarized by two deterministic quantities: a mean proxy $\mu_*$, asymptotically equal to $\mu_{\hat\theta} = \mathbb E[\hat\theta]$, and a fluctuation size $\alpha_*$, asymptotically given by $\alpha_*^2 \approx \mathrm{tr}(C_x C_{\hat\theta})$. These two quantities are obtained from an explicit min-max problem, or equivalently from the fixed-point system in Theorem 4.4.

   Under Assumption 4, Section 6 then shows that the test score behaves like
   $$
   x^\top \hat\theta  \approx  x^\top \mu_* + \alpha_* z,
   $$
   with $z \sim \mathcal N(0,1)$ independent of $(x,y)$. Hence score Gaussianity holds if and only if the informative projection $x^\top \mu_*$ is Gaussian.

   This matters in practice because a Gaussian proxy can miss skewness or bimodality of the score and therefore misestimate classification error. Our synthetic mixture and MNIST experiments illustrate this. At the same time, performance universality can still survive score non-universality when the metric depends only on low-order score moments rather than on the full score law; the ridge / squared-loss linear-model corollary is the clean example of this mechanism.

2. **Assumption 3**

   **Shared response B: Assumption 3**

   Assumption 3 does not require separability. It is a regularity condition on the third derivatives of $\rho$: it says that cross-coordinate third-order interactions are small enough at the $p=\Theta(n)$ fluctuation scale relevant to $\hat\theta - \mu_{\hat\theta}$. This is exactly the scale at which the quadratic surrogate is used, and it explains why only trace-level second-order quantities remain in the final formulas.

   The assumption is automatic for all quadratic regularizers, including anisotropic ridge, since $\nabla^3 \rho \equiv 0$, and for separable smooth penalties, since the off-diagonal part of the third-derivative tensor is zero. It also allows genuinely non-separable examples such as
   $$
   \rho(\theta) = \frac12\theta^\top H \theta + \lambda\phi\left(p^{-1/2} u^\top \theta\right),
   $$
   with $H \succeq \kappa I$, $\phi \in C^3$ with bounded third derivative, and $\|u\| = O(1)$. Here the extra third-order coupling is low-rank and satisfies the assumption.

   We will add such examples and clarify more explicitly why this is weaker than the standard weak-separability assumption: we control only the aggregate off-diagonal third-order effect, not a coordinate-wise decomposition.

3. We will also fix the typo `EMR` $\to$ `ERM`.










# Reviewer w5Yu

Summary:

This paper investigates high-dimensional convex empirical risk minimization to formalize when the widely used Gaussian universality assumption fails on non-Gaussian data. The authors heuristically extend the Convex Gaussian Min-Max Theorem through a quadratic surrogate for smooth regularizers, yielding deterministic equations for the estimator's asymptotic performance. Theoretically, they establish that a test score's Gaussianity hinges entirely on the projection of a test covariate onto an asymptotic signal proxy vector. Finally, experiments on synthetic and MNIST datasets validate this framework.
Strengths And Weaknesses:
Strengths

The authors investigate the ambitious goal of characterizing the breakdown of Gaussian universality, which is a fundamental question for the interpretation of asymptotics derived via AMP, RMT, and CGMT. The presentation is well-structured, with clear definitions of the three universality notions and careful positioning within the existing literature. The paper is overall technically sound and the results are interpretable. Theorems 3.3 and 7.1 add geometric intuition that simplifies the verification of universality in structured models. The empirical validation further corroborates the paper's claim.
Weaknesses

I have not identified any major weaknesses.

    Part of the results rely on the strong assumption of minimizer universality, which restricts the analysis to regimes where the estimators is sufficiently well-behaved. However, this limitation is transparently acknowledged and justified by the lack of known counterexamples.
    The font size in the figures' labels and legends is very small and hardly readable.

Soundness: 3: good
Presentation: 4: excellent
Significance: 3: good
Originality: 3: good
Key Questions For Authors:

    What are the precise conditions under which claim 4.1 could be made rigorous, or what are the fundamental obstacles?
    The author mentions that Assumption 3 is weaker that the standard weak separability assumption. Could you expand this discussion, providing examples on non-separable regularizers satisfying the assumption?
    Could you provide more general conditions allowing performance universality to survive the breakdown of score universality?


## Rebuttal

Thank you for the careful and positive reading. We will enlarge all figure labels and legends.

To make navigation easy, we use the same shared-response labels as in the other rebuttals.

1. **Claim 4.1 and the obstacles to a fully rigorous statement**

   **Shared response C: Claim 4.1 and Theorems 4.3--4.4**

   We agree that the draft should separate more clearly what is rigorous from what is conjectural. Claim 4.1 is meant to isolate the non-Gaussian extension of the CGMT comparison step; once that comparison principle is granted, the downstream convex analysis leading to Theorems 4.3 and 4.4 is explicit.

   A rigorous proof of Claim 4.1 would require a comparison theorem for the value and the optimizer of the non-Gaussian convex-concave saddle problem under the concentration and quadratic-growth conditions already isolated in the claim.

   The main obstacle is that standard CGMT relies on exact Gaussian rotational invariance to replace the bilinear form $\theta^\top A w$ by an auxiliary Gaussian process. For concentrated non-Gaussian columns, one needs a universality theorem for both the value and the optimizer of the saddle problem, together with control of the data-dependent shift $\mu_{\hat\theta}$. This is why we stated 4.1 as a Claim rather than a theorem.

   We will revise the wording so that Theorems 4.3 and 4.4 are explicitly presented as consequences of Claim 4.1, making the location of the heuristic step fully transparent.

2. **Assumption 3 and non-separable examples**

   Please see **Shared response B** in the rebuttals to Reviewers zqV1. In particular, we will add explicit non-separable examples such as
   $$
   \rho(\theta) = \frac12\theta^\top H \theta + \lambda\phi\left(p^{-1/2} u^\top \theta\right),
   $$
   and clarify why Assumption 3 is weaker than the usual weak-separability condition.

3. **When performance universality can survive score breakdown**

   A sufficient mechanism is that the target metric depends on the score only through low-order moments determined by $(\mu_*, \alpha_*)$, rather than through the full score law. The ridge / squared-loss linear-model corollary is the clean example of this phenomenon; please see **Shared response A** in the rebuttals to Reviewers zqV1 and zGDu.

   As a concrete empirical illustration, the anonymous repository includes a short notebook sweeping the elastic loss
   $$
   \mathcal L_\eta = (1-\eta)\ell_2 + \eta \ell_1
   $$
   on a bimodal model. At $\eta = 0$, squared loss shows performance universality even though the score is non-Gaussian; as $\eta$ increases toward the non-smooth $\ell_1$ regime, the Gaussian risk proxy degrades. We present this notebook only as an illustration of the boundary of the theory, not as a theorem beyond the smooth setting.













# Reviewer mdB9

Summary:

This paper studies high-dimensional empirical risk minimization (ERM) under non-Gaussian design and examines the limits of Gaussian universality, the practice of approximating non-Gaussian quantities (e.g., covariates, scores, or estimators) with the appropriate Gaussian ones. The authors analyze when such approximations hold and fail for convex ERM problems. Namely, building on ideas related to the Convex Gaussian Min–Max Theorem (CGMT), they propose a framework for non-Gaussian settings, under certain concentration assumptions, that yields a min–max characterization and fixed-point equations for approximating the mean and covariance of the ERM estimator, thereby identifying conditions under which Gaussian universality holds or breaks down. They further show that, under smoothness assumptions, a general regularizer can be asymptotically replaced by a quadratic surrogate defined by its gradient and Hessian, which simplifies the analysis. Numerical simulations are done to illustrate the theoretical predictions.
Strengths And Weaknesses:

Strengths

    The paper is clearly written.

    The problem is well-motivated, i.e. the question of when Gaussian universaility holds or breaks down is very important, and one of the results is veary elegant the test score is Gaussian if and only if the projection of covariates onto the estimator mean is Gaussian.

    The paper seems well situated within the existing literature and both extends and recovers several known results.

    The quadratic universality result is conceptually interesting and may have useful practical implications.

Weaknesses

    The technical contribution is not completely clear. Namely, the central CGMT extension, Claim 4.1, is not formally proven. Appendix C derives consequences but does not provide a rigorous proof of the claim itself. Since Theorems 4.3 and 4.4 rely on this claim, the lack of a proof or further comments on settings when it would hold weaken the theoretical contribution.

    The score characterization in Section 6 relies on Assumption 4 regarding minimizer universality, but there is lack of rigorous justification. As a result, one of the paper’s main results, Theorem 6.1 remains conditional. It would strengthen the paper to clarify when minimizer universality is expected to hold or fail, and prove/disprove in certain settings.

    The scope is also narrower than the title and abstract suggest. The assumptions require smoothness and strong convexity of the regularizer, with additional C^3-type control for the quadratic approximation argument. This excludes many practically important ERMs, especially non-smooth penalties such as lasso. So while the paper is about “general non-Gaussian designs,” it is not yet a broadly general ERM theory.

Soundness: 2: fair
Presentation: 3: good
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:

Questions

    My main question is can the authors identify specific technical reasons preventing a proof of Claim 4.1? Is it a conjectured extension, something provable under extra structural assumptions, or a reformulation of an existing result? Even partial results here would significantly strengthen the paper.
    Can the authors identify concrete model classes where Assumption 4 can be proved?
    Which parts of the analysis break down when the regularizer is non-smooth, e.g., lasso? Is the quadratic surrogate argument the main obstacle, or are there deeper issues?

Typos ecnountered

    Page 1: seminar work -> seminal work
    Page 2: relies some form ... -> relies on some form ...
    Page 8: he underlying heuristic - > the underlying heuristic
    Page 11: Proof of Subsection 4 results -> Proof of Subsection 3 results
    Page 14: Proof of Subsection 3 results -> Proof of Subsection 4 results

## Rebuttal

Thank you for pinpointing the main technical issues. We answer in the order of your questions and reuse the same labels as in the other rebuttals.

1. **Claim 4.1**

   Please see **Shared response C** in the rebuttals to Reviewers w5Yu. We agree that the paper should make this much more explicit: Claim 4.1 is the only non-rigorous comparison step, and Theorems 4.3 and 4.4 are downstream consequences of it. We will revise the wording accordingly.

2. **Assumption 4**

   **Shared response D: Assumption 4 / minimizer universality**

   We agree that Assumption 4 should be presented explicitly as a conditional hypothesis, not as an established theorem in the current draft. Its role is limited: it is used only in Section 6 to turn the mean/covariance characterization into a full score law; the non-Gaussian min-max / fixed-point characterization in Section 4 does not rely on it.

   The heuristic comes from the linearized KKT relation
   $$
   H_\rho(\hat\theta - \mu_{\hat\theta})
    \approx 
   -\nabla\rho(\mu_{\hat\theta})
   - \frac{1}{n}\sum_{i=1}^n \mathcal L'_{y_i}(x_i^\top \hat\theta)x_i.
   $$
   This suggests that $\hat\theta - \mu_{\hat\theta}$ behaves like an average of many weakly dependent terms. The difficulty is that the summands depend on $\hat\theta$ itself, so a standard CLT does not apply directly; one needs to control both the fluctuations of the optimizer and the dependence of the summands on that same optimizer.

   At present we do not claim a broad rigorous class where Assumption 4 is proved. Our point in the paper is to isolate it as the extra step needed for a full score law. We will revise the text to make this conditional status much more explicit.

3. **Nonsmooth penalties / scope**

   **Shared response E: nonsmooth penalties and scope**

   The limitation to smooth losses and smooth regularizers is broader than just the quadratic surrogate. Smoothness and strong convexity enter in three places:

   1. concentration and uniqueness of $\hat\theta$;
   2. the quadratic surrogate itself, via a second/third-order Taylor expansion;
   3. the linearized fluctuation heuristic behind Assumption 4.

   So the current paper is intentionally a smooth-ERM theory. This does not mean that a non-Gaussian CGMT cannot handle Lasso; it means that the present route is not the right one for Lasso. A different route, closer to direct nonsmooth CGMT arguments, would be needed.

   We will make this scope much more explicit in the limitations paragraph and in the abstract/introduction: concentrated designs, smooth losses, and smooth strongly convex regularizers.

4. Thank you for the typo list. We will correct all of them.



















# Reviewer zGDu

Summary:

This paper studies the asymptotic behavior of high-dimensional convex empirical risk minimization (ERM) under potentially non-Gaussian data designs satisfying a concentration assumption. Their goal is to characterize the distribution of the test score
and the resulting prediction performance in the proportional high-dimensional regime. The work focuses on convex losses and smooth regularizers, and develops a framework extending the CGMT approach beyond the standard Gaussian design setting.

The key idea is to derive deterministic equivalents for the mean and covariance of the ERM minimizer
, and use them to characterize the asymptotic distribution of the prediction score. The authors show that under concentration assumptions on the data and smoothness conditions on the loss and regularizer, the test score behaves asymptotically as
where and are solutions of deterministic and explicit saddle-point equations derived from the ERM objective. More precisely, the main results are:

    Quadratic universality of smooth regularizers: Any regularizer is asymptotically equivalent, for the statistical behavior of
    , to a quadratic surrogate determined by its Hessian at the origin and its gradient at the mean.
    Deterministic characterization via a min–max problem: The asymptotic statistics of
    are obtained from the solution of deterministic and explicit self-consistent equations which depends on the data distribution only through expectations involving the scalar variable .
    Performance universality in special cases: For ridge regression under a linear model, the asymptotic generalization error depends only on the first two moments of the features, implying performance universality even when the score distribution itself is non-Gaussian.

Moreover, the authors discuss conditions for score universality and a geometrical characterization for when Gaussian score universality is expected to hold. Overall, the paper provides a unified framework for analyzing high-dimensional ERM beyond the Gaussian design assumption. It clarifies the relationship between different notions of universality (minimizer, score, and performance) and identifies precise conditions under which Gaussian approximations for the prediction score are valid or break down.
Strengths And Weaknesses:

Strengths: Overall, the paper is clearly structured and the narrative is clear. The main results are also clearly highlighted. Universality in an important topic in the context of exact asymptotics, since provides stronger grounding for formulas which are typically derived under unrealistic data assumptions. Several papers have investigated this question over the past ten years, many of which at ICML. Therefore, it is a timely and relevant topic which is of interest to the exact asymptotics community at ICML. This paper provides some potentially interesting contributions in this directions, particularly in the relationship between different notions of universality appearing in the literature.

Weaknessess: In my reading, the three main weaknesses of this work are:

    Lack of discussion on the limitations. The paper makes a few broad assumptions, and from the reading it is unclear what are their scope. For example:
        In Assumption 1, the authors give an example where it holds. What about cases which it fails?
        Why is Claim 4.1 not a theorem? What are the challenges in proving it?
        The text suggests that Theorem 4.3 follows from Claim 4.1. Do you need to assume Claim 4.1 to hold in Theorem 4.3 or it is a particular case under which you can make it rigorous?
        The author says that assumption 3 does not hold for LASSO, but it holds for ridge. Are there other cases in which it holds? E.g. with ? Anisotropic regularization with satisfying some properties? From the final formulas, it seems that some delocalization is important for the universality result on the regularizer to hold, since it only depends on traces.
        The authors say they are not aware of any example where Assumption 4 does not hold. Do the authors believe this can be proven? What are the challenges in proving this?

    Lack of a concrete discussion of the main results. The paper is technically dense, and it would be useful to have some concrete discussion to stress the novelty of the results. For instance:
        Are there limits in which their formula reduce to the asymptotic Gaussian formula?
        The authors put forward that their result hold for non-Gaussian covariates and precisely characterize the break of universality. Although less common, non-Gaussian covariates were studied in different contexts already in the proportional regime in the literaure, e.g. single and multi-layer random features, Gaussian mixture, one-step of SGD, etc. For some of these examples, universality holds and for some it breaks. Do you have concrete examples covered by your results which are not discussed in the literature?

    Superficial discussion of the related literature. Although the authors make an effort to cite the universality literature (which I acknowledge is vast), the comparison with some of the closer works remain vague, which does not help understanding the novelty of the results. Also, some important references, both old and new, are missing. More precisely:
        Major:
        What the authors call "score universality"/1d CLT and its consequence for risk universality was first discussed in (Goldt et al., 2022), concurrently to (Hu & Lu 2022). In particular, both appeared before (Montanari & Saeed 2022), this should be more throughoutly acknowledged when referring to this connection.
        Discussion in L065-L072: The most general CGMT proof for convex ERM with Gaussian covariates appeared in (Loureiro et al., 2021b). This work also discussed empirically the limitations of these formulas, even in the case of unimodal distributions (e.g. see Fig. 3 therein)
        Different works have studied how to derive exact asymptotic results for problems in which Gaussian universality does not work, e.g. mixture distributions (Dandi et al., 2023), random features + one-step of SGD (Cui et al., 2024; Dandi et al., 2024), generalized linear models with quadratic features (Wen et al., 2025). A common denominator in these works is the so-called Conditional Gaussian Equivalence (cGET). This holds for problems in which the projection of the non-Gaussian component in the task on the data is low-dimensional, with the remainder of the randmness being enough to ensure concentratation in the limit. A deeper discussion on the relationship between cGET and the results in this work is pertinent, in particular with respect to Section 6, which seems to state a very similar condition.
        Minor:
        The discussion of Gaussian universality in the context of GMMs first appeared in (Gerace et al., 2022), before (Pesce et al., 2023).
        Discussion in L336-337: High-dimensional asymptotics for mixture models was also discussed in (Mignacco et al., 2020, Wang & Thrampoulidis 2020; Loureiro et al., 2021).

    (Goldt et al., 2022) The Gaussian equivalence of generative models for learning with shallow neural networks, https://proceedings.mlr.press/v145/goldt22a.html

    (Loureiro et al., 2021b) Learning curves of generic features maps for realistic datasets with a teacher-student model, https://proceedings.neurips.cc/paper/2021/hash/9704a4fc48ae88598dcbdcdf57f3fdef-Abstract.html

    (Gerace et al., 2022) Gaussian universality of perceptrons with random labels, https://arxiv.org/abs/2205.13303

    (Mignacco et al., 2020) The Role of Regularization in Classification of High-dimensional Noisy Gaussian Mixture, https://proceedings.mlr.press/v119/mignacco20a.html

    (Wang & Thrampoulidis 2020) Binary Classification of Gaussian Mixtures: Abundance of Support Vectors, Benign Overfitting and Regularization, https://arxiv.org/abs/2011.09148

    (Cui et al., 2024) Asymptotics of feature learning in two-layer networks after one gradient-step, https://arxiv.org/abs/2402.04980

    (Dandi et al., 2024) A Random Matrix Theory Perspective on the Spectrum of Learned Features and Asymptotic Generalization Capabilities, https://arxiv.org/abs/2410.18938

    (Wen et al., 2025) When does Gaussian equivalence fail and how to fix it: Non-universal behavior of random features with quadratic scaling, https://arxiv.org/abs/2512.03325

Soundness: 3: good
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:

Please refer to the questions in Weaknessess.


## Rebuttal

Thank you for the detailed review and for the very useful reference list.

To make navigation easy, we use the same shared-response labels as in the other rebuttals.

1. **Limitations and scope**

   We agree that the paper should state more plainly that Assumption 1 is a concentration assumption. It covers, for example, features of the form $x=\Phi(z)$ with Gaussian latent $z$ and Lipschitz $\Phi$, as well as class-conditional Lipschitz maps in classification, but it does not cover heavy-tailed or non-concentrated designs. We will add this explicitly.

   On Claim 4.1 and Assumptions 3/4, please see **Shared response C**, **Shared response B**, and **Shared response D** in the other rebuttals.

2. **Concrete discussion of the main results**

   Please see **Shared response A** in the rebuttals to Reviewers zqV1. We will also add two clarifications that would have helped the current draft:

   - In the Gaussian case, or more generally whenever $x^\top \mu_*$ is Gaussian, our score description reduces to the usual Gaussian fixed-point picture.
   - Beyond classical Gaussian design, Assumption 1 covers concentrated non-Gaussian representations such as Lipschitz feature maps of Gaussian latents. Our bimodal synthetic example and the MNIST experiment are concrete cases where the same $(\mu_*, \alpha_*)$ description remains predictive while the score itself is non-Gaussian.

   The anonymous repository linked in the paper reproduces these experiments and includes additional notebooks; we will point readers to it more clearly.

3. **Related work / chronology / cGET**

   We agree that the related-work section should be strengthened. We will explicitly acknowledge Goldt et al. (2022) together with Hu & Lu (2022) when discussing score universality / one-dimensional CLTs and their implications for performance universality, and clarify the chronology relative to Montanari & Saeed (2022). We will also expand the discussion around Loureiro et al. (2021b), Gerace et al. (2022), mixture-model work, and the more recent random-feature / cGET-related papers you listed.

   Regarding cGET, we agree that the connection is close in spirit. A useful way to state it is that cGET isolates cases where the non-Gaussian part of the problem lives in a low-dimensional informative component; our Section 6/7 viewpoint expresses a very similar phenomenon through the effective signal proxy $\mu_*$ and the informative subspace $F+\mathrm{span}(a)$. We will make this comparison explicit and clarify both the overlap and the difference in viewpoint.

4. We appreciate the references and will incorporate them.
