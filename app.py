import io
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Dashboard Deteksi Intrusi Jaringan - NSL-KDD",
    page_icon="🛡️",
    layout="wide",
)

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

ATTACK_MAPPING = {
    'normal': 'normal',
    # DoS - Denial of Service
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS',
    # Probe - Surveillance/scanning
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    # R2L - Remote to Local
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    # U2R - User to Root
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
}

ORDERS = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
COLORS = ['#3B82F6', '#EF4444', '#F59E0B', '#10B981', '#8B5CF6']

DEFAULT_DATA_PATH = "data/KDDTrain+.txt"


# =====================================================================
# FUNGSI BANTUAN (DI-CACHE AGAR TIDAK DIULANG SETIAP INTERAKSI)
# =====================================================================
@st.cache_data(show_spinner=False)
def load_data(file_bytes_or_path):
    """Memuat dataset NSL-KDD mentah dan menambahkan nama kolom."""
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        buf = io.BytesIO(file_bytes_or_path)
        df = pd.read_csv(buf, names=COLUMNS)
    else:
        df = pd.read_csv(file_bytes_or_path, names=COLUMNS)
    return df


@st.cache_data(show_spinner=False)
def preprocess(df):
    """Cek kualitas data, buang kolom difficulty, mapping label -> kategori."""
    missing = df.isnull().sum()
    duplicate = df.duplicated().sum()

    df = df.drop(columns=['difficulty'])
    df['attack_category'] = df['label'].map(ATTACK_MAPPING)
    df = df.drop(columns=['label'])

    return df, missing, duplicate


@st.cache_resource(show_spinner=False)
def train_all_models(df_hash_key, df, test_size, random_state):
    """Melatih 3 model dan mengembalikan hasil evaluasi + objek terlatih."""
    X = df.drop(columns=['attack_category'])
    y = df['attack_category']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    nominal_cols = ['protocol_type', 'service', 'flag']
    numeric_cols = [a for a in X.columns if a not in nominal_cols]

    preprocessor = make_column_transformer(
        (OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols),
        ('passthrough', numeric_cols)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    daftar_model = {
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=random_state),
    }

    hasil = {}
    for name, clf in daftar_model.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = (pred == y_test).mean()
        f1 = f1_score(y_test, pred, average='macro')
        recall = recall_score(y_test, pred, average='macro')
        hasil[name] = {
            'accuracy': acc, 'f1': f1, 'recall': recall,
            'pipeline': pipe, 'pred': pred,
        }

    model_terbaik = max(hasil, key=lambda k: hasil[k]['recall'])

    return {
        'hasil': hasil,
        'model_terbaik': model_terbaik,
        'le': le,
        'X_test': X_test,
        'y_test': y_test,
        'nominal_cols': nominal_cols,
        'numeric_cols': numeric_cols,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }


def fig_distribusi(counts):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=COLORS)
    ax.set_yscale('log')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f'{val:,}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Distribusi Serangan')
    ax.set_ylabel('Jumlah Serangan (skala log)')
    fig.tight_layout()
    return fig


def fig_confusion(y_test, pred, labels, title):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig, cm


def fig_feature_importance(pipeline, nominal_cols, numeric_cols, top_n=15):
    clf_step = pipeline.named_steps['classifier']
    pred_step = pipeline.named_steps['preprocessor']
    if not hasattr(clf_step, 'feature_importances_'):
        return None, None
    ohe_names = list(pred_step.named_transformers_['onehotencoder'].get_feature_names_out(nominal_cols))
    semua_fitur = ohe_names + numeric_cols
    fi_df = (pd.DataFrame({'fitur': semua_fitur, 'kepentingan': clf_step.feature_importances_})
             .sort_values('kepentingan', ascending=False).head(top_n))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=fi_df, x='kepentingan', y='fitur', hue='fitur', palette='viridis', legend=False, ax=ax)
    ax.set_title(f'Top {top_n} Feature Importance')
    fig.tight_layout()
    return fig, fi_df


# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.title("🛡️ Kontrol Dashboard")

uploaded = st.sidebar.file_uploader(
    "Upload dataset NSL-KDD (.txt / .csv tanpa header)",
    type=["txt", "csv"],
    help="Jika tidak upload, app akan mencoba memakai file di data/KDDTrain+.txt"
)

test_size = st.sidebar.slider("Proporsi data uji (test_size)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard ini melatih ulang 3 model (Decision Tree, Random Forest, "
    "Gradient Boosting) langsung di dalam app menggunakan pipeline yang sama "
    "dengan skrip `model.py` aslinya."
)

# =====================================================================
# LOAD DATA
# =====================================================================
try:
    if uploaded is not None:
        raw_df = load_data(uploaded.getvalue())
        sumber = uploaded.name
    else:
        raw_df = load_data(DEFAULT_DATA_PATH)
        sumber = DEFAULT_DATA_PATH
except FileNotFoundError:
    st.error(
        "Dataset belum tersedia. Silakan upload file NSL-KDD (mis. `KDDTrain+.txt`) "
        "lewat sidebar, atau letakkan file tersebut di folder `data/KDDTrain+.txt` "
        "pada repo sebelum deploy ke Streamlit."
    )
    st.stop()

df, missing, duplicate = preprocess(raw_df)

st.title("🛡️ Dashboard Deteksi Intrusi Jaringan — NSL-KDD")
st.caption(f"Sumber data: `{sumber}` | {len(raw_df):,} baris × {raw_df.shape[1]} kolom")

# =====================================================================
# TABS
# =====================================================================
tab_overview, tab_dist, tab_model, tab_cm, tab_fi, tab_predict = st.tabs([
    "📊 Ringkasan Data",
    "📈 Distribusi Serangan",
    "🤖 Perbandingan Model",
    "🧩 Confusion Matrix",
    "⭐ Feature Importance",
    "🔍 Coba Prediksi",
])

# ---------------------------------------------------------------------
# TAB 1: RINGKASAN DATA
# ---------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Baris", f"{len(raw_df):,}")
    col2.metric("Total Kolom (mentah)", raw_df.shape[1])
    col3.metric("Missing Value", int(missing.sum()))
    col4.metric("Baris Duplikat", int(duplicate))

    st.subheader("Pratinjau Data")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Tipe Data per Kolom")
    dtype_df = pd.DataFrame({
        'kolom': raw_df.columns,
        'dtype': raw_df.dtypes.astype(str),
        'missing': raw_df.isnull().sum().values,
    })
    st.dataframe(dtype_df, use_container_width=True, height=300)

    with st.expander("Lihat detail missing value per kolom"):
        st.dataframe(missing[missing > 0] if missing.sum() > 0 else
                     pd.Series({'info': 'Tidak ada missing value 🎉'}))

# ---------------------------------------------------------------------
# TAB 2: DISTRIBUSI SERANGAN
# ---------------------------------------------------------------------
with tab_dist:
    counts = df['attack_category'].value_counts().reindex(ORDERS)
    pct = (counts / counts.sum() * 100).round(2)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.pyplot(fig_distribusi(counts))
    with c2:
        st.subheader("Tabel Distribusi")
        tabel = pd.DataFrame({
            'Kategori': counts.index,
            'Jumlah': counts.values,
            'Persentase (%)': pct.values
        })
        st.dataframe(tabel, use_container_width=True, hide_index=True)
        st.info(
            "Dataset NSL-KDD sangat **tidak seimbang** — kategori `U2R` dan `R2L` "
            "jumlahnya jauh lebih kecil dibanding `normal`/`DoS`. Karena itu evaluasi "
            "model di tab berikutnya menonjolkan **recall macro**, bukan hanya accuracy."
        )

# ---------------------------------------------------------------------
# LATIH MODEL (dipakai oleh beberapa tab di bawah)
# ---------------------------------------------------------------------
with st.spinner("Melatih model (Decision Tree, Random Forest, Gradient Boosting)..."):
    train_out = train_all_models(len(df), df, test_size, int(random_state))

hasil = train_out['hasil']
model_terbaik = train_out['model_terbaik']
le = train_out['le']
X_test = train_out['X_test']
y_test = train_out['y_test']
nominal_cols = train_out['nominal_cols']
numeric_cols = train_out['numeric_cols']

# ---------------------------------------------------------------------
# TAB 3: PERBANDINGAN MODEL
# ---------------------------------------------------------------------
with tab_model:
    st.subheader("Hasil Pembagian Data")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah Data Latih", f"{train_out['n_train']:,}")
    c2.metric("Jumlah Data Uji", f"{train_out['n_test']:,}")

    st.subheader("Perbandingan Metrik 3 Model")
    metrik_df = pd.DataFrame({
        name: {
            'Accuracy': d['accuracy'],
            'F1-macro': d['f1'],
            'Recall-macro': d['recall'],
        } for name, d in hasil.items()
    }).T.round(4)
    st.dataframe(
        metrik_df.style.highlight_max(axis=0, color='#16a34a33'),
        use_container_width=True
    )

    st.bar_chart(metrik_df)

    st.success(f"🏆 Model terbaik berdasarkan **Recall macro** adalah **{model_terbaik}**")

# ---------------------------------------------------------------------
# TAB 4: CONFUSION MATRIX
# ---------------------------------------------------------------------
with tab_cm:
    pilihan_model = st.selectbox("Pilih model untuk dilihat confusion matrix-nya", list(hasil.keys()),
                                  index=list(hasil.keys()).index(model_terbaik))
    data_model = hasil[pilihan_model]
    fig_cm, cm = fig_confusion(
        y_test, data_model['pred'], le.classes_,
        f"{pilihan_model} | Acc={data_model['accuracy']:.2%} | Recall={data_model['recall']:.2%}"
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(fig_cm)
    with c2:
        st.subheader("Recall per Kategori")
        per_class_recall = cm.diagonal() / cm.sum(axis=1)
        recall_df = pd.DataFrame({
            'Kategori': le.classes_,
            'Recall': per_class_recall
        }).sort_values('Recall')
        st.dataframe(recall_df.style.format({'Recall': '{:.2%}'}), use_container_width=True, hide_index=True)
        st.caption(
            "Kategori dengan recall paling rendah biasanya `U2R`/`R2L` karena "
            "jumlah sampelnya sangat sedikit di dataset."
        )

# ---------------------------------------------------------------------
# TAB 5: FEATURE IMPORTANCE
# ---------------------------------------------------------------------
with tab_fi:
    pilihan_model_fi = st.selectbox(
        "Pilih model untuk feature importance", list(hasil.keys()),
        index=list(hasil.keys()).index(model_terbaik), key="fi_select"
    )
    top_n = st.slider("Jumlah fitur teratas ditampilkan", 5, 25, 15)
    fig_fi, fi_df = fig_feature_importance(
        hasil[pilihan_model_fi]['pipeline'], nominal_cols, numeric_cols, top_n
    )
    if fig_fi is None:
        st.warning(f"Model **{pilihan_model_fi}** tidak memiliki atribut `feature_importances_`.")
    else:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.pyplot(fig_fi)
        with c2:
            st.dataframe(fi_df.reset_index(drop=True), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 6: COBA PREDIKSI
# ---------------------------------------------------------------------
with tab_predict:
    st.subheader("Uji Model dengan Sampel dari Data Uji")
    pilihan_model_pred = st.selectbox(
        "Pilih model untuk prediksi", list(hasil.keys()),
        index=list(hasil.keys()).index(model_terbaik), key="pred_select"
    )
    pipe_terpilih = hasil[pilihan_model_pred]['pipeline']

    idx = st.number_input(
        "Pilih indeks baris dari data uji (0 s.d. {})".format(len(X_test) - 1),
        min_value=0, max_value=len(X_test) - 1, value=0, step=1
    )
    sample = X_test.iloc[[idx]]
    actual_label = le.inverse_transform([y_test[idx]])[0]
    pred_label = le.inverse_transform(pipe_terpilih.predict(sample))[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Label Sebenarnya", actual_label)
    c2.metric("Prediksi Model", pred_label)
    c3.metric("Status", "✅ Benar" if actual_label == pred_label else "❌ Salah")

    with st.expander("Lihat detail fitur baris ini"):
        st.dataframe(sample.T.rename(columns={sample.index[0]: 'nilai'}), use_container_width=True)

    st.markdown("---")
    st.subheader("Prediksi Massal dari File CSV")
    st.caption(
        "Upload file CSV berisi kolom-kolom fitur (tanpa kolom `label`/`attack_category`) "
        "untuk diprediksi sekaligus menggunakan model terpilih di atas."
    )
    batch_file = st.file_uploader("Upload CSV untuk prediksi massal", type=["csv"], key="batch_predict")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            preds = pipe_terpilih.predict(batch_df)
            batch_df['prediksi_kategori'] = le.inverse_transform(preds)
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "⬇️ Unduh Hasil Prediksi (CSV)",
                data=batch_df.to_csv(index=False).encode('utf-8'),
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

# =====================================================================
# DOWNLOAD MODEL TERLATIH
# =====================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Unduh Model")
buffer = io.BytesIO()
joblib.dump({'pipeline': hasil[model_terbaik]['pipeline'], 'label_encoder': le}, buffer)
st.sidebar.download_button(
    label=f"Unduh model_nsl.pkl ({model_terbaik})",
    data=buffer.getvalue(),
    file_name="model_nsl.pkl",
    mime="application/octet-stream",
)
