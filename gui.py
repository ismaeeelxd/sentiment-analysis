from pathlib import Path
import pickle
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets/frame0") 
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)
window = Tk()
window.title("Sentiment Analysis")
window.geometry("799x697")
window.configure(bg = "#FFFFFF")
canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 697,
    width = 799,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    406.0,
    651.0,
    image=image_image_1
)

canvas.create_text(
    293.0,
    640.0,
    anchor="nw",
    text="Prediction",
    fill="#030202",
    font=("MontserratRoman ExtraBold", 12 * -1)
)

predict_text_id = canvas.create_text(
    375.0,
    643.0,
    anchor="nw",
    text="predict",
    fill="#030202",
    font=("MontserratRoman SemiBold", 12 * -1)
)

canvas.create_text(
    142.0,
    115.0,
    anchor="nw",
    text="Text:",
    fill="#CB4B4B",
    font=("MontserratRoman SemiBold", 20 * -1)
)

# --- Load vectorizer and models ---
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

model_files = [
    './saved_models/model_linear_svm.pkl',
    './saved_models/model_svm_(sigmoid).pkl',
    './saved_models/model_naive_bayes.pkl',
    './saved_models/model_logistic_regression.pkl',
    './saved_models/model_random_forest.pkl'
]
models = []
for file in model_files:
    with open(file, 'rb') as f:
        models.append(pickle.load(f))

# --- Entry box setup ---
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    413.5,
    294.0,
    image=entry_image_1
)
entry_1 = Text(
    bd=0,
    bg="#35498F",
    fg="#000716",
    highlightthickness=0,
    wrap="word"
)
entry_1.place(
    x=90.0,
    y=147.0,
    width=647.0,
    height=292.0
)

# --- Checkbox setup ---
checkbox_states = [False] * 5
checkbox_coords = [
    (120, 470),  # Linear SVM
    (258, 470),  # SVM (Sigmoid)
    (395, 470),  # Naive Bayes
    (533, 470),  # Logistic Regression
    (671, 470),  # Random Forest
]
model_labels = [
    "Linear SVM", "    SVM \n(Sigmoid)", " Naive \n Bayes", "   Logistic \nRegression", "Random Forest"
]
checkbox_ids = []
selected_model_idx = [None]  # Use a list for mutability

def toggle_checkbox(idx):
    for i in range(len(checkbox_states)):
        checkbox_states[i] = False
        canvas.itemconfig(checkbox_ids[i], fill="#FFFFFF")
    checkbox_states[idx] = True
    canvas.itemconfig(checkbox_ids[idx], fill="#CB4B4B")
    selected_model_idx[0] = idx

for i, (x, y) in enumerate(checkbox_coords):
    rect = canvas.create_rectangle(x, y, x+20, y+20, fill="#FFFFFF", outline="#CB4B4B", width=2)
    checkbox_ids.append(rect)
    canvas.tag_bind(rect, "<Button-1>", lambda e, idx=i: toggle_checkbox(idx))
    canvas.create_text(x+10, y+35, text=model_labels[i], fill="#CB4B4B", font=("MontserratRoman SemiBold", 14), anchor="n")

# --- Prediction display label ---
prediction_label = canvas.create_text(
    400, 600,  # Centered at bottom
    text="Prediction: ",
    fill="#CB4B4B",
    font=("MontserratRoman SemiBold", 16)
)

# --- Predict button and logic ---
def predict_from_entry():
    idx = selected_model_idx[0]
    if idx is None:
        canvas.itemconfig(predict_text_id, text="Please select a model.")
        return
    text = entry_1.get("1.0", "end-1c")
    if not text.strip():
        canvas.itemconfig(predict_text_id, text="Please enter some text.")
        return
    X_test = vectorizer.transform([text])
    model = models[idx]
    y_pred = model.predict(X_test)
    if y_pred[0] == 0:
        result = "Negative"
    else:
        result = "Positive"
    canvas.itemconfig(predict_text_id, text=f"{result}")

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=predict_from_entry,
    relief="flat"
)
button_1.place(
    x=334.0,
    y=571.0,
    width=144.0,
    height=50.0
)

canvas.create_text(
    300.0,
    25.0,
    anchor="nw",
    text="Sentiment analysis",
    fill="#030202",
    font=("Pacifico Regular", 32 * -1)
)


icon_sentiment_image = PhotoImage(file=relative_to_assets("sentiment-analysis.png"))
canvas.create_image(
    190.0, 0.0, 
    image=icon_sentiment_image,
    anchor="nw"
)


icon_text_image = PhotoImage(file=relative_to_assets("file.png"))
canvas.create_image(
    95.0, 95.0,  
    image=icon_text_image,
    anchor="nw"
)

window.resizable(False, False)
window.mainloop()
