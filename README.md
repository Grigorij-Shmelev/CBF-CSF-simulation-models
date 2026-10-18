### Model Comparison: Cerebrovascular Hemodynamics vs. Craniospinal CSF Hydrodynamics
**A Mathematical & Clinical Comparison of Ursino (2010), Štěpáník (2009), and Vaněk (2015) Models**

*Prepared for Bachelor Thesis*

--------------------------------------------------------------------------------

#### Part 1: Ursino & Giannessi (2010) Model
##### Core Biophysical Purpose
*   **Primary Domain:** Regional Cerebral Blood Flow (CBF), Circle of Willis collateral redistribution, active autoregulation, and $\mathrm{CO}_2$ reactivity.
*   **State Vector:** 23 Ordinary Differential Equations (ODEs).
##### Key Architectural Features
1.  **Anatomical Grid:**
    *   6 regional brain territories: Left/Right Middle ($ml, mr$), Anterior ($al, ar$), and Posterior ($pl, pr$) Cerebral Arteries.
    *   Circle of Willis collateral bridges: Anterior Communicating Artery ($\mathrm{ACoA}$) and Posterior Communicating Arteries ($\mathrm{PCoA}$).
    *   Distal cortical anastomoses ($R_{c,ams}, R_{c,pms}, R_{c,aa}, R_{c,pp}$).
2.  **Active Control Mechanisms:**
    *   **Autoregulation ($x_{\mathrm{aut}}$):** 1st-order low-pass filter ($\tau = 20\text{ s}$) driving vasomotion based on relative flow error $\mathrm{rel\_q} = (q - q_n)/q_n$.
    *   **$\mathrm{CO}_2$ Reactivity ($x_{\mathrm{CO}_2}$):** Vasodilation/vasoconstriction response ($\tau = 40\text{ s}$) driven by $P_a\mathrm{CO}_2$ fluctuations.
    *   **Ischemic Acidosis Suppression ($A_{\mathrm{CO}_2}$):** Sigmoidal switch suppressing $\mathrm{CO}_2$ reactivity when flow drops below $-50\%$ of normal ($\mathrm{rel\_q} < -0.5, b_{\mathrm{CO}_2} = -0.5$).
3.  **Active Resistance & Compliance Coupling:**

$$C_d(x) = C_{dn} \cdot \left[ \left(1 - \frac{\Delta C}{2}\right) + \frac{(1 + \frac{\Delta C}{2}) e^{x/k_{Cd}}}{1 + e^{x/k_{Cd}}} \right]$$

$$R_d = k_R \cdot \frac{C_{dn}^2}{V_d^2}$$

##### Key Simulation Benchmark: Unilateral ICA Occlusion & Steal Phenomenon
*   **Baseline Left MCA Flow:** $3.75\text{ mL/s}$
*   **100% Left ICA Occlusion (Pre-Hypercapnia):** Drops to $1.53\text{ mL/s}$ (Ischemic territory maximally dilates via autoregulation).
*   **Hypercapnia Challenge ($P_a\mathrm{CO}_2 = 60\text{ mmHg}$):**
    *   **Healthy Right MCA Flow:** Surges from $3.79\text{ mL/s} \rightarrow 5.77\text{ mL/s}$ (+52%).
    *   **Ischemic Left MCA Flow:** Drops from $1.53\text{ mL/s} \rightarrow 1.43\text{ mL/s}$ (**Steal Phenomenon confirmed**).
    *   **Mechanism:** Vasodilation on the healthy side lowers Circle of Willis perfusion pressure, actively stealing blood away from the ischemic territory.

--------------------------------------------------------------------------------

#### Part 2: Zbyšek Štěpáník (2009) Model
##### Core Biophysical Purpose
*   **Primary Domain:** Craniospinal CSF transport, pressure distribution across 5 anatomical compartments, and lumbar infusion test diagnostics.
*   **State Vector:** 5 Differential Equations for relative pressure differentials.
##### Compartmental Layout
1.  **Compartment 1:** Ventricles ($P_1$)
2.  **Compartment 2:** Intracranial Subarachnoid Space ($P_2 / \mathrm{ICP}$)
3.  **Compartment 3:** Cervical Subarachnoid Space ($P_3$)
4.  **Compartment 4:** Thoracic Subarachnoid Space ($P_4$)
5.  **Compartment 5:** Lumbar Subarachnoid Space ($P_5$)
##### Architectural Characteristics & Native Boundaries ($P_V = 2.0\text{ mmHg}$)
*   **Single Boundary Interface:** Driven primarily by an arterial boundary ($a$) with baseline compliance $C_{2A0} = 0.20\text{ mL/mmHg}$.
*   **Compressed Venous Bed:** At native $P_V = 2.0\text{ mmHg}$, the transmural pressure $p_{2V} = P_2 - P_V = 8.0\text{ mmHg}$ compresses venous compliance down to $0.074\text{ mL/mmHg}$.
*   **Pressure Elevation:** Reaches a high 5-minute peak ICP of **$21.05\text{ mmHg}$ ($286.2\text{ mmH}_2\mathrm{O}$)** (and up to $24.43\text{ mmHg} / 332.3\text{ mmH}_2\mathrm{O}$ at full steady state), yielding a calculated outflow resistance $R_{\mathrm{out}} \approx 7.38\text{ mmHg/(mL/min)}$.

--------------------------------------------------------------------------------

#### Part 3: Petr Vaněk (2015) 3rd Generation Model
##### Core Biophysical Purpose
*   **Primary Domain:** Extended craniospinal CSF transport, multi-frequency (cardiac + respiratory) boundary forcing, and clinical long-term ICP recording.
*   **State Vector:** 5 Differential Equations with explicit dual arterial/venous driving terms.
##### Structural Upgrades Over Štěpáník (2009)
1.  **Explicit Intracranial Venous Sub-Compartment ($b$ / mozkové žíly):**
    *   Štěpáník's model used a single arterial interface ($a$). Vaněk explicitly split the boundary into arterial ($a$) and venous ($b$) sub-compartments.
    *   Added venous compliances: $C_{1b0} = 0.60\text{ mL/mmHg}$ (Ventricles) and $C_{2b0} = 0.60\text{ mL/mmHg}$ (Intracranial SAS).
2.  **Physiological Compliance Adjustment:**
    *   Reduced maximum arterial compliance by 10x ($C_{2a0} = 0.02\text{ mL/mmHg}$ vs. Štěpáník's $0.20\text{ mL/mmHg}$) to reflect realistic arterial stiffness.
    *   Shifted ~85–90% of spatial volume buffering onto the highly compressible venous bed ($C_{2b0} = 0.60\text{ mL/mmHg}$).
3.  **Damped Pressure Derivative & Time Constant:**

$$\frac{dP_2}{dt} = \frac{dV_2/dt + C_{2a}\frac{dP_a}{dt} + C_{2b}\frac{dP_b}{dt}}{C_{2a} + C_{2b}}$$

    *   The compressible venous cushion increases the overall system time constant ($\tau = R \cdot C$), preventing artificial pressure spikes during fluid infusion.
##### Lumbar Infusion Test Results ($P_b = 10.0\text{ mmHg}$)
*   **5-Minute Peak ICP ($P_2$):** Reaches **$17.49\text{ mmHg}$ ($237.9\text{ mmH}_2\mathrm{O}$)**.
*   **Damping Effect:** Shows a ~$48.3\text{ mmH}_2\mathrm{O}$ lower pressure elevation after 5 minutes (and up to ~$104\text{ mmH}_2\text{O}$ lower vs. Štěpáník's steady-state peak) due to venous cushioning.

--------------------------------------------------------------------------------

#### Part 4: Quantitative Side-by-Side Comparison
##### Lumbar Infusion Test Metrics ($1.5\text{ mL/min}$ for 300 seconds, Native Boundaries)
| Metric / Parameter | Štěpáník (2009) | Vaněk (2015) | Difference / Biophysical Impact |
| ------ | ------ | ------ | ------ |
| **Intracranial Boundary** | Single Arterial ($a$) | Dual Arterial ($a$) + Venous ($b$) | Venous bed added for vascular cushioning |
| **Native Boundary Pressure** | $P_V = 2.0\text{ mmHg}$ | $P_b = 10.0\text{ mmHg}$ | Native baseline venous drainage pressure |
| **Max Arterial Compliance ($C_{2a0}$)** | $0.20\text{ mL/mmHg}$ | $0.02\text{ mL/mmHg}$ | 10x reduction reflecting arterial stiffness |
| **Max Venous Compliance ($C_{2b0}$)** | Unformalized ($0.60$) | $0.60\text{ mL/mmHg}$ | Handles 85–90% of intracranial volume buffering |
| **Baseline ICP ($P_{2,0}$)** | $9.98\text{ mmHg}$ | $9.96\text{ mmHg}$ | Normal baseline resting pressure |
| **5-Min Peak Infusion ICP ($P_{2,\text{inf}}$)** | **$21.05\text{ mmHg}$** | **$17.49\text{ mmHg}$** | **$3.56\text{ mmHg}$ lower in Vaněk** |
| **5-Min Peak ICP in $\text{mmH}_2\mathrm{O}$** | **$286.2\text{ mmH}_2\mathrm{O}$** | **$237.9\text{ mmH}_2\mathrm{O}$** | **$48.3\text{ mmH}_2\mathrm{O}$ lower in Vaněk** |
| **Full Steady-State Peak ICP** | **$24.43\text{ mmHg}$ ($332.3\text{ mmH}_2\mathrm{O}$)** | **$23.40\text{ mmHg}$ ($318.2\text{ mmH}_2\text{O}$)** | Štěpáník climbs much faster to peak |
| **ICP Elevation ($\Delta P$)** | $11.07\text{ mmHg}$ | $7.53\text{ mmHg}$ | Vaněk dampens transient pressure rise |
| **Apparent $R_{\text{out}}$ at $t=5\text{ min}$** | $7.38\text{ mmHg/(mL/min)}$ | $5.02\text{ mmHg/(mL/min)}$ | Longer time constant delays plateau in Vaněk |

--------------------------------------------------------------------------------

#### Part 5: Comprehensive 3-Model Summary Matrix
| Feature / Dimension | Ursino & Giannessi (2010) | Štěpáník (2009) | Petr Vaněk (2015) |
| ------ | ------ | ------ | ------ |
| **Primary Domain** | **Cerebral Hemodynamics & Blood Flow** | **Craniospinal CSF Hydrodynamics** | **Craniospinal CSF Hydrodynamics** |
| **Anatomical Scope** | 6 Regional Territories + Circle of Willis | 5 Spatial Compartments (Ventricles to Lumbar) | 5 Spatial Compartments + Dual Vascular Boundaries |
| **Active Regulation** | **Autoregulation & $\mathrm{CO}_2$ Reactivity** | None (Passive tissue compliances) | None (Passive non-linear compliances) |
| **Vascular Mechanics** | Active $R_d(x)$ & $C_d(x)$ state functions | Single arterial boundary ($a$) | Dual arterial ($a$) + venous ($b$) sub-compartments |
| **Key Clinical Phenomenon** | Hypercapnia, Steal, Carotid Stenosis | Lumbar Infusion Test, $R_{\text{out}}$ | Lumbar Infusion Test, Venous Cushioning |
| **Suitability for RL Agent** | **Primary Engine** (Native control variables) | Benchmark model | Benchmark model |

--------------------------------------------------------------------------------

#### Specific Questions
1.  **Data Formats for Validation:** 
*	For the MRI project, what specific clinical data channels will be provided for model validation? Are we targeting Phase-Contrast MRI flow velocities (mL/s) in the internal carotid arteries, or time-resolved ventricular volume changes?*
2. **Sampling Frequency &amp; Time Grid:**
*	"Our differential equation solver (* *solve\_ivp* *) outputs data on an* **adaptive time grid** *with sub-millisecond precision during pressure transients. Does your PC-MRI processing pipeline prefer our raw adaptive time steps, or should we export the CSV resampled to a fixed frequency (e.g., 50 Hz or 100 Hz) matching your scanner's cardiac cine frame rate?"*
3. **Target Patient Parameters:**
*	"Is primary goal to overlay our simulated $F_{23}$ flow curves onto patient MRI scans to estimate non-invasive craniospinal compliance ($C = \frac{dV}{dP}$) and outflow resistance ($R_{\mathrm{out}}$)?"
