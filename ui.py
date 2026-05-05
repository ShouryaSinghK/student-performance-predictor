import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from classifier import train_model

model, scaler, train_acc, test_acc, k_scores, cm, report = train_model()

best_k = 4
MAX_STUDY_HOURS = 80

def predict():
    try:
        attendance = float(entry_attendance.get())
        marks = float(entry_marks.get())
        assignments = float(entry_assignments.get())
        hours = float(entry_hours.get())
        backlogs = float(entry_backlogs.get())

        if any(v < 0 for v in [attendance, marks, assignments, hours, backlogs]):
            output_label.config(text="Values cannot be negative", fg="orange")
            root.update()
            return
        if attendance > 100:
            output_label.config(text="Attendance cannot exceed 100%", fg="orange")
            root.update()
            return
        if marks > 100:
            output_label.config(text="Marks cannot exceed 100", fg="orange")
            root.update()
            return
        if assignments > 100:
            output_label.config(text="Assignments cannot exceed 100%", fg="orange")
            root.update()
            return
        if hours > MAX_STUDY_HOURS:
            output_label.config(text=f"Study hours cannot exceed {MAX_STUDY_HOURS}hrs/week", fg="orange")
            root.update()
            return

        features = np.array([[attendance, marks, assignments, hours, backlogs]])
        scaled = scaler.transform(features)
        result = model.predict(scaled)
        output_label.config(text="Result: PASS ✓" if result[0]==1 else "Result: FAIL ✗",
                           fg="green" if result[0]==1 else "red")
        root.update()

    except ValueError:
        output_label.config(text="Please enter valid numbers", fg="orange")
        root.update()

def reset():
    entry_attendance.delete(0, tk.END)
    entry_marks.delete(0, tk.END)
    entry_assignments.delete(0, tk.END)
    entry_hours.delete(0, tk.END)
    entry_backlogs.delete(0, tk.END)
    output_label.config(text="")
    root.update()

def open_analysis():
    win = tk.Toplevel(root)
    win.title("Model Analysis")
    win.geometry("700x750")

    tk.Label(win, text="K Value vs Accuracy", font=("Arial", 13, "bold")).pack(pady=10)

    # Table
    frame = tk.Frame(win)
    frame.pack()

    tk.Label(frame, text="K Value", width=12, font=("Arial", 10, "bold"),
             bg="#333333", fg="white", relief="ridge").grid(row=0, column=0)
    tk.Label(frame, text="Test Accuracy %", width=16, font=("Arial", 10, "bold"),
             bg="#333333", fg="white", relief="ridge").grid(row=0, column=1)

    best_acc = max(k_scores.values())
    for i, (k_val, acc) in enumerate(k_scores.items(), start=1):
        is_best = (k_val == best_k)
        bg = "#90EE90" if is_best else "white"
        font_style = ("Arial", 10, "bold") if is_best else ("Arial", 10)
        tk.Label(frame, text=f"k = {k_val}", width=12,
                 bg=bg, font=font_style, relief="ridge").grid(row=i, column=0)
        tk.Label(frame, text=f"{acc}%", width=16,
                 bg=bg, font=font_style, relief="ridge").grid(row=i, column=1)

    tk.Label(win, text=f"★ Best k = {best_k} highlighted in green",
             font=("Arial", 9), fg="gray").pack(pady=4)

    # Confusion Matrix
    tk.Label(win, text="Confusion Matrix (k=4)", font=("Arial", 13, "bold")).pack(pady=8)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Classification Report
    tk.Label(win, text="Classification Report", font=("Arial", 13, "bold")).pack(pady=8)
    tk.Label(win, text=report, font=("Courier", 9), justify="left",
             bg="#f0f0f0", relief="groove", padx=10, pady=8).pack(padx=20)

# Main window
root = tk.Tk()
root.title("Student Performance Predictor")
root.geometry("500x550")

tk.Label(root, text="Student Performance Predictor", font=("Arial", 14, "bold")).pack(pady=10)
tk.Label(root, text=f"Model: KNN  |  Best k={best_k}  |  Train Acc: {train_acc*100:.1f}%  |  Test Acc: {test_acc*100:.1f}%",
         font=("Arial", 9), fg="gray").pack(pady=2)

tk.Label(root, text="Attendance %").pack()
entry_attendance = tk.Entry(root)
entry_attendance.pack()

tk.Label(root, text="Average Marks").pack()
entry_marks = tk.Entry(root)
entry_marks.pack()

tk.Label(root, text="Assignments Completed %").pack()
entry_assignments = tk.Entry(root)
entry_assignments.pack()

tk.Label(root, text=f"Study Hours per Week (max {MAX_STUDY_HOURS})").pack()
entry_hours = tk.Entry(root)
entry_hours.pack()

tk.Label(root, text="Backlogs").pack()
entry_backlogs = tk.Entry(root)
entry_backlogs.pack()

tk.Button(root, text="Predict", command=predict, bg="blue", fg="white").pack(pady=15)
tk.Button(root, text="Reset", command=reset, bg="gray", fg="white").pack(pady=5)
tk.Button(root, text="View Model Analysis", command=open_analysis,
          bg="#333333", fg="white").pack(pady=8)

output_label = tk.Label(root, text="", font=("Arial", 13, "bold"))
output_label.pack()

root.lift()
root.focus_force()
root.mainloop()