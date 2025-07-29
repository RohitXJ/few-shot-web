# Few-Shot Learning Model Service (Web Version)

This is a web-based few-shot learning model builder, powered by [Streamlit](https://streamlit.io/) and connected to our custom [few\_shot\_lib](https://github.com/RohitXJ/few_shot_lib) framework. It allows users to upload images, train a model using meta-learning, and download the final PyTorch `.pt` model file—all through a clean and interactive interface.

👉 **GitHub Repository**: [RohitXJ/few-shot-web](https://github.com/RohitXJ/few-shot-web)

---

## 🔍 What is Few-Shot Learning?

Few-shot learning is a type of meta-learning that allows models to generalize well even with very few examples per class. This is particularly useful in scenarios where collecting a large amount of data is not feasible. Instead of traditional training, we use support and query image sets to fine-tune and evaluate models in a few-shot learning setting.

This project allows you to train such models instantly using your own data.

---

## 🎯 Features

* ✅ Upload support and query images for 2–10 classes
* ✅ Automatically balanced validation and evaluation logic
* ✅ Choose from multiple lightweight CNN backbones
* ✅ Uses our custom [few\_shot\_lib](https://github.com/RohitXJ/few_shot_lib) engine
* ✅ Downloads final model for reuse or deployment

---

## 🧠 Available Backbone Models

The app uses one of the following pretrained models as the feature extractor:

| Model                | Size (MB) | Notes                    |
| -------------------- | --------- | ------------------------ |
| resnet18             | 45 MB     | Small, reliable          |
| resnet34             | 83 MB     | Medium size              |
| resnet50             | 98 MB     | Heavier, better accuracy |
| mobilenet\_v2        | 14 MB     | Lightweight mobile model |
| mobilenet\_v3\_small | 10 MB     | Extra small              |
| mobilenet\_v3\_large | 16 MB     | Balanced                 |
| efficientnet\_b0     | 20 MB     | Efficient and compact    |
| efficientnet\_b1     | 32 MB     | Slightly larger variant  |
| densenet121          | 33 MB     | Deeper connections       |
| densenet169          | 57 MB     | Bigger DenseNet          |

---

## ⚙️ How It Works

1. **Upload Images:**

   * Choose number of classes (2–10)
   * Upload support and query images for each class

2. **Select Backbone:**

   * Pick one model backbone from the dropdown

3. **Train & Evaluate:**

   * Backend runs few-shot training pipeline
   * Outputs accuracy, label predictions
   * Exports trained model (`.pt`) for download

4. **Download Model:**

   * Exported model is usable in any PyTorch project

---

## 📦 Use Trained Model in PyTorch

You can directly use the trained `.pt` model from this web app in any Python project using our `few_shot_lib` package.

### 🔧 Install the library

```bash
pip install fewshotlib
```

### 📚 Project link

[https://github.com/RohitXJ/few\_shot\_lib](https://github.com/RohitXJ/few_shot_lib)

This gives you full flexibility to evaluate or fine-tune the model in your own PyTorch environment.

---

## ⚠️ Real-World Disclaimer

This app is intended for experimental and educational purposes. Due to the nature of few-shot learning:

* ❌ Accuracy is not guaranteed in real-world noisy data
* ⚠️ Small datasets are more prone to overfitting
* ✅ Balanced support and query sets are required
* ⛔ Do not expect 100% accuracy in all cases

---

## 🚀 Deployment

You can deploy this project using:

* [Streamlit Community Cloud](https://streamlit.io/cloud)
* Local hosting: `streamlit run app.py`
* Containerized: via Docker (optional)

---

## 🔒 Security Notes

* All uploads are stored temporarily in the `data_temp/` directory
* Inputs are validated to prevent training errors
* Filenames and paths are sanitized before saving

---

## 👨‍💻 Developed By

**Rohit Gomes**
Connect with me on [LinkedIn](https://www.linkedin.com/in/rohit-gomes-12209620a)
📦 Main Library: [few\_shot\_lib](https://github.com/RohitXJ/few_shot_lib)
🌐 Web UI: [few-shot-web](https://github.com/RohitXJ/few-shot-web)

---

## 📜 License

This project is under a **custom license**. It is intended for **educational and research purposes only**.

**Commercial use is strictly prohibited.**
