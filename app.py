from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

pd.options.mode.chained_assignment = None

# Membaca data
data = dataframe = pd.read_csv(r'lungCancer.csv', delimiter=';')

# Seleksi kolom yang akan digunakan
data = dataframe[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']]

# Mengubah kolom 'LUNG_CANCER' dan 'GENDER' menjadi numerik
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
data['GENDER'] = data['GENDER'].map({'M': 0, 'F': 1})

# Ubah nilai kolom menjadi biner (1 dan 0)
binary_columns = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                'CHRONIC_DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 
                'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
                'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

# Ganti nilai 2 menjadi 1 dan nilai 1 menjadi 0
for col in binary_columns:
    data[col] = data[col].apply(lambda x: 1 if x > 1 else 0)

# Normalisasi kolom AGE menggunakan metode z-score
scaler = preprocessing.StandardScaler()
data['AGE'] = scaler.fit_transform(data[['AGE']])  # Hanya menormalisasi kolom AGE

# Grouping variabel untuk fitur dan label
X = data[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
           'CHRONIC_DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL_CONSUMING', 
           'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']].values  
y = data['LUNG_CANCER'].values  # Label

# Pembagian training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# permodelan Random Forest
random_forest = RandomForestClassifier(random_state=0 )
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)



@app.route('/')
def index():
        return render_template('index.html')
    
    
@app.route('/result', methods=['POST'])
def result():
    try:
        # Mengambil data dari form
        name = request.form.get('name')
        age = request.form.get('usia')

        # Mengambil gejala yang dipilih
        gender = request.form.get('gender')
        smoking = request.form.get('smoking')
        yellow_fingers = request.form.get('yellow_fingers')
        anxiety = request.form.get('anxiety')
        peer_pressure = request.form.get('peer_pressure')
        chronic_disease = request.form.get('chronic_disease')
        fatigue = request.form.get('fatigue')
        allergy = request.form.get('allergy')
        wheezing = request.form.get('wheezing')
        alcohol_consuming = request.form.get('alcohol_consuming')
        coughing = request.form.get('coughing')
        shortness_of_breath = request.form.get('shortness_of_breath')
        swallowing_difficulty = request.form.get('swallowing_difficulty')
        chest_pain = request.form.get('chest_pain')

        # Konversi input menjadi numerik sesuai preprocessing
        gender_num = 0 if gender == 'male' else 1
        smoking_num = 1 if smoking == 'yes' else 0
        yellow_fingers_num = 1 if yellow_fingers == 'yes' else 0
        anxiety_num = 1 if anxiety == 'yes' else 0
        peer_pressure_num = 1 if peer_pressure == 'yes' else 0
        chronic_disease_num = 1 if chronic_disease == 'yes' else 0
        fatigue_num = 1 if fatigue == 'yes' else 0
        allergy_num = 1 if allergy == 'yes' else 0
        wheezing_num = 1 if wheezing == 'yes' else 0
        alcohol_consuming_num = 1 if alcohol_consuming == 'yes' else 0
        coughing_num = 1 if coughing == 'yes' else 0
        shortness_of_breath_num = 1 if shortness_of_breath == 'yes' else 0
        swallowing_difficulty_num = 1 if swallowing_difficulty == 'yes' else 0
        chest_pain_num = 1 if chest_pain == 'yes' else 0

        # Normalisasi usia menggunakan scaler yang telah dilatih
        age_normalized = scaler.transform([[age]])[0][0]

        # Menyiapkan input untuk prediksi
        input_data = np.array([[gender_num, age_normalized, smoking_num, yellow_fingers_num, anxiety_num, 
                                peer_pressure_num, chronic_disease_num, fatigue_num, allergy_num, wheezing_num,
                                alcohol_consuming_num, coughing_num, shortness_of_breath_num, 
                                swallowing_difficulty_num, chest_pain_num]])

        # Melakukan prediksi
        prediction = random_forest.predict(input_data)[0]

        # Menentukan hasil
        diagnosis_result = "Anda kemungkinan mengidap kanker paru-paru." if prediction == 1 else "Kemungkinan Anda tidak mengidap kanker paru-paru."

        # Mengumpulkan gejala yang dipilih untuk ditampilkan
        selected_symptoms = []
        symptoms = {
            'Merokok': 'Ya' if smoking_num == 1 else 'Tidak',
            'Jari Kekuningan': 'Ya' if yellow_fingers_num == 1 else 'Tidak',
            'Kecemasan': 'Ya' if anxiety_num == 1 else 'Tidak',
            'Tekanan Lingkungan': 'Ya' if peer_pressure_num == 1 else 'Tidak',
            'Penyakit Kronis': 'Ya' if chronic_disease_num == 1 else 'Tidak',
            'Kelelahan': 'Ya' if fatigue_num == 1 else 'Tidak',
            'Alergi': 'Ya' if allergy_num == 1 else 'Tidak',
            'Mengi': 'Ya' if wheezing_num == 1 else 'Tidak',
            'Konsumsi Alkohol': 'Ya' if alcohol_consuming_num == 1 else 'Tidak',
            'Batuk': 'Ya' if coughing_num == 1 else 'Tidak',
            'Sesak Napas': 'Ya' if shortness_of_breath_num == 1 else 'Tidak',
            'Kesulitan Menelan': 'Ya' if swallowing_difficulty_num == 1 else 'Tidak',
            'Nyeri Dada': 'Ya' if chest_pain_num == 1 else 'Tidak'
        }

        for symptom, value in symptoms.items():
            selected_symptoms.append(f"{symptom}: {value}")

        selected_symptoms_str = "\n".join(selected_symptoms)

        # Menentukan jenis kelamin
        gender_str = 'Laki-laki' if gender_num == 0 else 'Perempuan'

        # (Opsional) Menambahkan saran perawatan berdasarkan hasil diagnosis
        suggested_treatment = "Disarankan untuk berkonsultasi dengan dokter spesialis paru-paru untuk tindakan lebih lanjut."

        return render_template('result.html',
                               user_name=name,
                               user_age=age,
                               user_gender=gender_str,
                               selected_symptoms=selected_symptoms_str,
                               diagnosis_result=diagnosis_result,
                               suggested_treatment=suggested_treatment)
    except Exception as e:
        return render_template('result.html', error=str(e))


    
if __name__ == '__main__':
    app.run(debug=True)