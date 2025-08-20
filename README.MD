## 🤖 Introduction Algorithm Decision Tree
Hi everyone, today i make decision tree from scratch with CPP language. So i want explain how works this code or decision tree algorithm. So this is a some of part decision tree:
1.Leaf Node: Starting point representing the whole dataset.
2.Branches: Lines connecting nodes showing the flow from one decision to another.
3.Internal Nodes: Points where decisions are made based on data features.
4.Leaf Nodes: End points of the tree where the final decision or prediction is made.

## 📌 Ilustrasition
This is a ilustrasition for you and get imagine how works decision tree:

![Decision Tree Illustration](https://media.geeksforgeeks.org/wp-content/uploads/20250626155729954380/Decision-tree-1.webp)

This ilustrasition from geeks for geeks.

## ⚠️ Disclaimer 
If you running again and again my project, you get a change for my accuracy, MSE, and feature prediction for target. I get accuracy, MSE, and feature after run my project like this:

=== Improvement Model dengan Backward Pass ===
Melakukan backward pass untuk improve model...
MSE: 7.74368e+06
R2_SCORE: 0.845848

=== Contoh Prediksi Single Sample ===
Sample pertama dari test set:
Fitur: [0.195652, 1, 0.449556, 0, 1]
Prediksi: 2790.53
Actual: 2497.04
Error: 293

The accuracy and MSE are good because I applied MinMax scaling to the feature variables (X), not to the target variable (y), since the y variable represents the actual prediction values and after prediction get a prediction like actual dataset not scaler data (range 0-1 because im use minmaxscaler).

## 🚀 How to running project
For running my project, don't forget to like this:
```bash
# Open your terminal, you must use Windows Power Shell because this instruction running project, i use windows power shell for running my project.
# Compile program, you must in the folder src not global!
g++ main.cpp -o main
# Running program
./main.exe
```
## 👋 Closer
*I hope you enjoy read my explain about decision tree and you get a new knowledge about how works decision tree. So thank you and good bye!*

![Anime girl closing image](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/13070a77-c3e7-4960-9621-d99d604bf731/dh2iolx-04947b22-e8b2-46b7-b167-d001c15fb7aa.jpg/v1/fill/w_1280,h_906,q_75,strp/sayonara_by_23wazzz_dh2iolx-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9OTA2IiwicGF0aCI6IlwvZlwvMTMwNzBhNzctYzNlNy00OTYwLTk2MjEtZDk5ZDYwNGJmNzMxXC9kaDJpb2x4LTA0OTQ3YjIyLWU4YjItNDZiNy1iMTY3LWQwMDFjMTVmYjdhYS5qcGciLCJ3aWR0aCI6Ijw9MTI4MCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.FnmtS8MXW7KJN_Ho1KP-7hJIg9qyaDulrNfZQATf3e0)

Art from Deviant art
---
