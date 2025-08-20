## 🤖 Introduction Algorithm Decision Tree
Hi everyone, today i make decision tree from scratch with CPP language. So i want explain how works this code or decision tree algorithm. So this is a some of part decision tree:
1.Leaf Node: Starting point representing the whole dataset.
2.Branches: Lines connecting nodes showing the flow from one decision to another.
3.Internal Nodes: Points where decisions are made based on data features.
4.Leaf Nodes: End points of the tree where the final decision or prediction is made.

## 📌 Ilustrasition
This is a ilustrasition for you and get imagine how works decision tree:

![](path/to/inserted-image-or-the-url-will-render-as-gambar-di-README)

This ilustrasition from geeks for geeks.

## ⚠️ Disclaimer 
If you running again and again my project, you get a change for my accuracy, MSE, and feature prediction for target. I get accuracy, MSE, and feature after run my project like this:

=== Improvement Model dengan Backward Pass ===
Melakukan backward pass untuk improve model...
MSE: 1.0048e+07
R2_SCORE: 0.810766

=== Contoh Prediksi Single Sample ===
Sample pertama dari test set:
Fitur: [0.5, 0, 0.541297, 0.2, 1]
Prediksi: 7730.05
Actual: 6781.35
Error: 948

The accuracy and MSE are good because I applied MinMax scaling to the feature variables (X), not to the target variable (y), since the y variable represents the actual prediction values and after prediction get a prediction like actual dataset not scaler data (range 0-1 because im use minmaxscaler).

## 🚀 How to running project
For running my project, don't forget to like this:
```bash
# Open your terminal, you must use Windows Power Shell because this instruction running project, i use windows power shell for running my project.
# Compile program, you must in the folder src not global!
g++ main.cpp -o main
# Running program
./main.exe

## 👋 Closer
*I hope you enjoy read my explain about decision tree and you get a new knowledge about how works decision tree. So thank you and good bye!*

<img src="images/anime_closing.png" alt="Anime girl closing image" align="center" width="300"/>

---