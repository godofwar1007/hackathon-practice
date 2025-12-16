# Beatles Quantum Classifier (QHack Challenge)

## 📌 Project Overview
This project solves a binary classification problem using a **Quantum K-Nearest Neighbors (QKNN)** algorithm.

The objective is to determine if a new person likes The Beatles ("YES" or "NO") based on two classical features:
1.  **Age**
2.  **Minutes spent watching TV**

The solution leverages quantum computing principles—specifically the **SWAP Test**—to calculate the similarity (distance) between different people's data vectors on a quantum processor.

---

## 🧠 Key Concepts

To solve this problem, we utilized three core quantum computing concepts:

### 1. Amplitude Embedding
Since quantum computers operate on qubits, we cannot feed classical numbers (like "Age: 20") directly into the circuit. We used **Amplitude Embedding** to map classical data vectors into the amplitudes of a quantum state.
* **Function Used:** `qml.AmplitudeEmbedding`
* **Normalization:** Quantum states must be unit vectors (length = 1). We utilized `normalize=True` to automatically scale the raw input data (e.g., `[20, 100]`) into valid quantum states without manual preprocessing.

### 2. The SWAP Test
The SWAP Test is a famous quantum algorithm used to measure the overlap (similarity) between two quantum states, $|A\rangle$ and $|B\rangle$.
* **The Circuit Logic:**
    1.  **Hadamard Gate ($H$):** Puts the "Ancilla" qubit (Wire 0) into superposition.
    2.  **Controlled-SWAP ($CSWAP$):** Swaps the states of Person A (Wire 1) and Person B (Wire 2) *only if* the Ancilla is $|1\rangle$.
    3.  **Hadamard Gate ($H$):** Closes the interference loop on the Ancilla.
* **The Result:** If the two states are identical, the Ancilla will always measure as 0. If they are different, the probability of measuring 0 decreases.

### 3. Probability Measurement
We strictly used `qml.probs(wires=0)` instead of `expval` or `sample`.
* **Reason:** The mathematical formula for the SWAP test relies specifically on $P(0)$—the probability of the Ancilla being in the $|0\rangle$ state. `qml.probs` provides this exact value directly.

---

## 🛠️ Code Construction (Step-by-Step)

Here is how the solution code was built in `solution.py`:

1.  **Device Setup**:
    We initialized a `default.qubit` device with **3 wires**:
    * `Wire 0`: The Ancilla (helper) qubit for the test.
    * `Wire 1`: Encodes Person A's data.
    * `Wire 2`: Encodes Person B's data.

2.  **Data Loading**:
    Inside the QNode, we applied `AmplitudeEmbedding` to Wires 1 and 2. We ensured `pad_with=0.` was set to handle data safety and `normalize=True` to handle the vector math.

3.  **Circuit Execution**:
    We applied the **H $\rightarrow$ CSWAP $\rightarrow$ H** sequence.

4.  **Math & Post-Processing**:
    The quantum circuit returns the probability $P(0)$. We then wrote classical Python code to convert this probability into the **Euclidean Distance**, which the K-Nearest Neighbors algorithm requires to vote on the final "YES/NO" label.

---

## 📐 Mathematical Formulas

The code implements the following derivation to convert quantum measurements into classical distance:

**1. Calculate Squared Overlap**
First, we recover the overlap from the Ancilla's probability $P(0)$:
$$|\langle A | B \rangle|^2 = 2 \cdot P(0) - 1$$

**2. Calculate Euclidean Distance**
Finally, we convert the overlap into the distance metric required by the classifier:
$$Distance = \sqrt{2 \cdot (1 - |\langle A | B \rangle|)}$$

---

## 📦 Requirements
* Python 3.x
* PennyLane (`pip install pennylane`)
* NumPy
