import sys
import pennylane as qml
from pennylane import numpy as np



def distance(A, B):
    """Function that returns the distance between two vectors.

    Args:
        - A (list[int]): person's information: [age, minutes spent watching TV].
        - B (list[int]): person's information: [age, minutes spent watching TV].

    Returns:
        - (float): distance between the two feature vectors.
    """
    dev=qml.device('default.qubit',wires=3)

    @qml.qnode(dev)
    def swap_test_circuit(vec_a,vec_b):
        
        # loading the data 
        qml.AmplitudeEmbedding(features=vec_a, wires=1, pad_with=0., normalize=True)
        qml.AmplitudeEmbedding(features=vec_b, wires=2, pad_with=0., normalize=True)

        # the swap test algorithm 
        qml.Hadamard(wires=0)       # putting the ancilla qubit in superposition 
        qml.CSWAP(wires=[0, 1, 2]) 
        qml.Hadamard(wires=0)       

        return qml.probs(wires=0)  # measurement of the probability of ancilla being 0
    
    probs = swap_test_circuit(A, B)
    
    prob_0 = probs[0]  # the probability of measuing 0

    overlap_squared = 2 * prob_0 - 1  # the calculation of the squared overlap: |<A|B>|^2=2*P(0)-1

   
    if overlap_squared<0:  # counter measure for negative squared overlap due to tiny floating point errors 
        overlap_squared=0

    overlap = np.sqrt(overlap_squared)
    
    dist = np.sqrt(2 * (1 - overlap))
    
    return float(dist)    




    # QHACK #

    # The Swap test is a method that allows you to calculate |<A|B>|^2 , you could use it to help you.
    # The qml.AmplitudeEmbedding operator could help you too.

    # dev = qml.device("default.qubit", ...
    # @qml.qnode(dev)

    # QHACK #


def predict(dataset, new, k):
    """Function that given a dataset, determines if a new person do like Beatles or not.

    Args:
        - dataset (list): List with the age, minutes that different people watch TV, and if they like Beatles.
        - new (list(int)): Age and TV minutes of the person we want to classify.
        - k (int): number of nearby neighbors to be taken into account.

    Returns:
        - (str): "YES" if they like Beatles, "NO" otherwise.
    """

    # DO NOT MODIFY anything in this code block

    def k_nearest_classes():
        """Function that returns a list of k near neighbors."""
        distances = []
        for data in dataset:
            distances.append(distance(data[0], new))
        nearest = []
        for _ in range(k):
            indx = np.argmin(distances)
            nearest.append(indx)
            distances[indx] += 2

        return [dataset[i][1] for i in nearest]

    output = k_nearest_classes()

    return (
        "YES" if len([i for i in output if i == "YES"]) > len(output) / 2 else "NO",
        float(distance(dataset[0][0], new)),
    )


# QHack main execution block 
"""

if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    dataset = []
    new = [int(inputs[0]), int(inputs[1])]
    k = int(inputs[2])
    for i in range(3, len(inputs), 3):
        dataset.append([[int(inputs[i + 0]), int(inputs[i + 1])], str(inputs[i + 2])])

    output = predict(dataset, new, k)
    sol = 0 if output[0] == "YES" else 1
    print(f"{sol},{output[1]}")


""" 

# main exectution block for the demo (non interactive)
"""



if __name__ == "__main__":

    dataset = [
        [[55, 10], "NO"],    # Person A (Old, Low TV) -> Hates Beatles
        [[60, 100], "YES"],  # Person B (Old, High TV) -> Loves Beatles
        [[19, 15], "NO"],    # Person C (Young, Low TV) -> Hates Beatles
        [[25, 120], "YES"],  # Person D (Young, High TV) -> Loves Beatles
        [[22, 95], "YES"],   # Person E (Young, High TV) -> Loves Beatles
        [[52, 8], "NO"]      # Person F (Old, Low TV) -> Hates Beatles
    ]

    
    new_age = 23
    new_tv = 90
    new_point = [new_age, new_tv]
    
    k_value = 3 # Look at 3 nearest neighbors

    # --- THE TERMINAL DISPLAY ---
    print("\n" + "="*60)
    print("      QUANTUM BEATLES CLASSIFIER (QML-200)")
    print("="*60)
    print(f" [+] Loaded Training Data: {len(dataset)} samples")
    print(f" [+] Test Subject: Age {new_age}, TV Time {new_tv} mins")
    print("-" * 60)
    print(" [>] Initializing Quantum Circuit (3 Qubits)...")
    print(" [>] Encoding Classical Data into Quantum Amplitudes...")
    print(f" [>] Running SWAP Tests to find {k_value} nearest neighbors...")
    
    result = predict(dataset, new_point, k_value) 
    
    prediction_label = result[0] 
    
    print("-" * 60)
    print(f"   PREDICTION: {prediction_label}")
    print("-" * 60)
    
    if prediction_label == "YES":
        print(" >> CONCLUSION: This person is a BEATLES FAN.")
    else:
        print(" >> CONCLUSION: This person is NOT a fan.")
    print("="*60 + "\n")
    
"""
# main exectution block - interactive at the terminal

if __name__ == "__main__":
   
    dataset = [
        [[55, 10], "NO"],    # Person A
        [[60, 100], "YES"],  # Person B
        [[19, 15], "NO"],    # Person C
        [[25, 120], "YES"],  # Person D
        [[22, 95], "YES"],   # Person E
        [[52, 8], "NO"]      # Person F
    ]
    k_value = 3

    print("\n" + "="*50)
    print("  QUANTUM BEATLES CLASSIFIER (LIVE)  ")
    print("="*50)
    print(f" [+] System Ready. Database size: {len(dataset)}")
    print("-" * 50)
    
    try:
     
        print("Please enter the Test Subject details:")
        input_age = int(input("   >> Enter Age (e.g., 25): "))
        input_tv  = int(input("   >> Enter TV Minutes (e.g., 100): "))
        
        new_point = [input_age, input_tv]
        
        print("-" * 50)
        print(f" [>] Encoding values {new_point} into Qubits...")
        print(" [>] Running Quantum SWAP Tests...")

       
        result = predict(dataset, new_point, k_value)
        
        label = result[0] if isinstance(result, (list, tuple)) else result

        print("-" * 50)
        print(f"  PREDICTION: {label}")
        
        if label == "YES":
            print(" >> Verdict: This person LOVES The Beatles!")
        else:
            print(" >> Verdict: This person is NOT a fan.")
            
    except ValueError:
        print("\n [!] Error: Please enter valid integer numbers only!")
        
    print("="*50 + "\n")
