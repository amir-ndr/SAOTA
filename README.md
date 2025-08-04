# SAOTA: Semi-Asynchronous Federated Learning via Over-the-Air Aggregation

This repository contains the implementation of **SAOTA**, a novel **Semi-Asynchronous Federated Learning (FL)** framework that leverages **Over-the-Air (OTA)** analog aggregation and **Lyapunov-based optimization** for efficient model training in wireless networks.

> 📄 **Paper**: *Learning on the Air: Semi-Asynchronous Federated Learning via Over-the-Air Computation*  
> 🎓 Developed by **Amirmohammad Ramezannaderi** @ *Queen's University*

---

## 🌟 Key Features

- **Semi-asynchronous client scheduling**  
  Balances straggler mitigation and stale model impact.
  
- **OTA aggregation**  
  Reduces communication latency using simultaneous analog transmission.
  
- **Energy & time constrained optimization**  
  Achieves efficient training using Lyapunov drift-plus-penalty control.
  
- **Closed-form power allocation**  
  Adapts to gradient norms, channel gain, and energy queues.
  
- **Virtual-queue-based client selection**  
  Dynamically selects clients based on real-time system state and constraints.

---

## 🛠️ Code Structure
.
├── client.py # Client-side training, gradient, and energy tracking

├── server.py # Server logic: client selection, OTA aggregation, optimization

├── main.py # Simulation and evaluation script

├── model.py # CNN model used for training on MNIST

├── dataloader.py # Data loading and Dirichlet partitioning

├── fl_system.log # Logging file generated during run

├── semi_async_ota_fl_results.png # Final results visualization



