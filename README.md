
## SPATL-XLC: An Explainability-Driven Framework for Efficient and Robust Federated Learning under Non-IID Data

Federated learning (FL) enables multiple devices to collectively train a machine learning
model without sharing private data. However, when data across devices differ significantly (non-IID),training
becomes less accurate and difficult to understand in terms of how the model makes decisions (explainability).
To address this problem, we propose two methods: SPATL-XL and SPATL-XLC. SPATL-XL applies SHAP,
a model explainability technique, to identify and remove less important model parameters before sending updates to the server. SPATL-XLC builds on this by clustering similar clients based on their similarity patterns
and sending the aggregated cluster updates, thus improving learning stability and transparency. Experiments
on public datasets (CIFAR-10 and Fashion MNIST) show that SPATL-XLC improves explanation fidelity
from 30% to 89% and reduces the training time by 13% while maintaining practical accuracy. Although
the communication cost slightly increases, the method delivers a clearer and more stable explanation with a
reduced per-round bandwidth, making it well-suited for real-world FL applications in which explainability
and efficiency are critical.


<img width="727" height="560" alt="image" src="https://github.com/user-attachments/assets/0fbe053a-d067-4858-8a43-7d2174f8cd20" />
