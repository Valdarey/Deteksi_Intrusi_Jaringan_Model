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
def load_data():
    """Memuat dataset NSL-KDD dari file lokal."""
    try:
        df = pd.read_csv(DEFAULT_DATA_PATH, names=COLUMNS)
        return df
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan! Pastikan file 'KDDTrain+.txt' ada di folder 'data/'")
        st.stop()


@st.cache_data(show_spinner=False)
def preprocess(df):
    """Cek kualitas data, buang kolom difficulty, mapping label -> kategori."""
    missing = df.isnull().sum()
    duplicate = df.duplicated().sum()

    # Simpan label asli untuk referensi
    original_labels = df['label'].copy()
    
    df = df.drop(columns=['difficulty'])
    df['attack_category'] = df['label'].map(ATTACK_MAPPING)
    df = df.drop(columns=['label'])

    return df, missing, duplicate, original_labels


@st.cache_resource(show_spinner=False)
def train_all_models(df, test_size, random_state):
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


@st.cache_data(show_spinner=False)
def get_label_counts(df, original_labels):
    """Mendapatkan distribusi label dari data asli"""
    counts = original_labels.value_counts()
    return counts


def fig_distribusi(counts):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=COLORS[:len(counts)])
    ax.set_yscale('log')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f'{val:,}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Distribusi Serangan (Label Asli)')
    ax.set_ylabel('Jumlah Serangan (skala log)')
    ax.set_xlabel('Label Serangan')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return fig


def fig_distribusi_kategori(df):
    """Distribusi per kategori serangan"""
    counts = df['attack_category'].value_counts().reindex(ORDERS)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=COLORS)
    ax.set_yscale('log')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f'{val:,}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Distribusi Kategori Serangan')
    ax.set_ylabel('Jumlah (skala log)')
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


def fig_protocol_distribution(df):
    """Visualisasi distribusi protokol"""
    fig, ax = plt.subplots(figsize=(6, 4))
    protocol_counts = df['protocol_type'].value_counts()
    ax.pie(protocol_counts.values, labels=protocol_counts.index, autopct='%1.1f%%', 
           colors=['#3B82F6', '#EF4444', '#10B981'], startangle=90)
    ax.set_title('Distribusi Protokol')
    fig.tight_layout()
    return fig


def fig_service_top(df, top_n=10):
    """Visualisasi top service"""
    fig, ax = plt.subplots(figsize=(8, 5))
    service_counts = df['service'].value_counts().head(top_n)
    bars = ax.barh(service_counts.index, service_counts.values, color='#636EFA')
    ax.set_title(f'Top {top_n} Service')
    ax.set_xlabel('Jumlah')
    for bar, val in zip(bars, service_counts.values):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:,}', 
                va='center', ha='left', fontsize=9)
    fig.tight_layout()
    return fig


def fig_flag_distribution(df):
    """Visualisasi distribusi flag"""
    fig, ax = plt.subplots(figsize=(7, 4))
    flag_counts = df['flag'].value_counts()
    bars = ax.bar(flag_counts.index, flag_counts.values, color='#F59E0B')
    ax.set_title('Distribusi Flag')
    ax.set_ylabel('Jumlah')
    for bar, val in zip(bars, flag_counts.values):
        if val > 1000:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{val:,}', ha='center', fontsize=8)
    fig.tight_layout()
    return fig


# =====================================================================
# LOAD DATA
# =====================================================================
raw_df = load_data()
df, missing, duplicate, original_labels = preprocess(raw_df)

# =====================================================================
# SIDEBAR - Kontrol Dashboard
# =====================================================================
st.sidebar.title("🛡️ Kontrol Dashboard")
st.sidebar.markdown("---")

test_size = st.sidebar.slider("Proporsi data uji (test_size)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard ini menggunakan dataset NSL-KDD yang sudah tersedia. "
    "Data: 125.973 baris × 42 kolom."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistik Cepat")
st.sidebar.metric("Total Baris", f"{len(raw_df):,}")
st.sidebar.metric("Tipe Serangan", raw_df['label'].nunique())
st.sidebar.metric("Normal vs Attack", 
                  f"{len(df[df['attack_category']=='normal']):,} / {len(df[df['attack_category']!='normal']):,}")

st.sidebar.markdown("---")
st.sidebar.info("⏱️ **Pelatihan Model**")
st.sidebar.caption(
    "Proses pelatihan 3 model (Decision Tree, Random Forest, Gradient Boosting) "
    "memakan waktu sekitar **1-2 menit** tergantung spesifikasi server."
)

# =====================================================================
# MAIN CONTENT
# =====================================================================
st.title("🛡️ Dashboard Deteksi Intrusi Jaringan — NSL-KDD")
st.caption(f"Data: NSL-KDD Train+ | {len(raw_df):,} baris × {raw_df.shape[1]} kolom")

# Tampilkan informasi waktu pelatihan di awal
st.info("⏱️ **Proses pelatihan model sedang berlangsung...**\n\n"
        "Pelatihan 3 model (Decision Tree, Random Forest, Gradient Boosting) "
        "memakan waktu sekitar **1-2 menit**. Harap tunggu hingga selesai.\n\n"
        "💡 *Hasil pelatihan akan di-cache sehingga tidak perlu dilatih ulang "
        "saat navigasi tab.*")

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
    col2.metric("Total Kolom", raw_df.shape[1])
    col3.metric("Missing Value", int(missing.sum()))
    col4.metric("Baris Duplikat", int(duplicate))
    
    st.markdown("---")
    
    # Visualisasi ringkasan
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Protokol")
        st.pyplot(fig_protocol_distribution(df))
    
    with col2:
        st.subheader("📊 Distribusi Flag (Top)")
        st.pyplot(fig_flag_distribution(df))
    
    st.subheader("📊 Top 10 Service")
    st.pyplot(fig_service_top(df, 10))
    
    st.subheader("Pratinjau Data")
    st.dataframe(raw_df.head(10), use_container_width=True)
    
    with st.expander("Lihat Tipe Data per Kolom"):
        dtype_df = pd.DataFrame({
            'kolom': raw_df.columns,
            'dtype': raw_df.dtypes.astype(str),
            'missing': raw_df.isnull().sum().values,
        })
        st.dataframe(dtype_df, use_container_width=True, height=300)

# ---------------------------------------------------------------------
# TAB 2: DISTRIBUSI SERANGAN
# ---------------------------------------------------------------------
with tab_dist:
    st.subheader("📊 Distribusi Kategori Serangan")
    
    # Kategori counts
    category_counts = df['attack_category'].value_counts().reindex(ORDERS)
    cat_pct = (category_counts / category_counts.sum() * 100).round(2)
    
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.pyplot(fig_distribusi_kategori(df))
    with c2:
        st.subheader("Tabel Kategori")
        tabel_cat = pd.DataFrame({
            'Kategori': category_counts.index,
            'Jumlah': category_counts.values,
            'Persentase (%)': cat_pct.values
        })
        st.dataframe(tabel_cat, use_container_width=True, hide_index=True)
        st.info(
            "📌 Dataset NSL-KDD sangat **tidak seimbang** — kategori `U2R` dan `R2L` "
            "jumlahnya jauh lebih kecil dibanding `normal`/`DoS`."
        )
    
    st.markdown("---")
    st.subheader("📊 Distribusi Label Asli (Detail)")
    
    # Gunakan original_labels untuk distribusi label asli
    label_counts = original_labels.value_counts()
    label_pct = (label_counts / label_counts.sum() * 100).round(2)
    
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.pyplot(fig_distribusi(label_counts))
    with c2:
        st.subheader("Tabel Label Asli")
        tabel_label = pd.DataFrame({
            'Label': label_counts.index,
            'Jumlah': label_counts.values,
            'Persentase (%)': label_pct.values
        })
        st.dataframe(tabel_label, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Detail Label per Kategori")
    
    # Filter by category
    categories = st.multiselect(
        "Pilih Kategori untuk Detail",
        options=ORDERS,
        default=['DoS', 'normal', 'Probe']
    )
    
    if categories:
        # Filter original labels based on category
        filtered_labels = [k for k, v in ATTACK_MAPPING.items() if v in categories]
        detail_mask = original_labels.isin(filtered_labels)
        detail_counts = original_labels[detail_mask].value_counts()
        
        if len(detail_counts) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(detail_counts.index, detail_counts.values, color='#636EFA')
            ax.set_title('Detail Label per Kategori yang Dipilih')
            ax.set_ylabel('Jumlah')
            ax.set_xlabel('Label')
            for bar, val in zip(bars, detail_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                        f'{val:,}', ha='center', fontsize=8)
            plt.xticks(rotation=45, ha='right')
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Tidak ada label untuk kategori yang dipilih.")

# ---------------------------------------------------------------------
# LATIH MODEL (dipakai oleh beberapa tab di bawah)
# ---------------------------------------------------------------------
with st.spinner("⏳ Melatih model (Decision Tree, Random Forest, Gradient Boosting)...\n\n⏱️ Proses ini memakan waktu sekitar 1-2 menit..."):
    train_out = train_all_models(df, test_size, int(random_state))

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
    st.success("✅ Model berhasil dilatih!")
    st.caption("⏱️ Waktu pelatihan: sekitar 1-2 menit (tergantung spesifikasi server)")
    
    st.subheader("📊 Hasil Pembagian Data")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah Data Latih", f"{train_out['n_train']:,}")
    c2.metric("Jumlah Data Uji", f"{train_out['n_test']:,}")

    st.subheader("📊 Perbandingan Metrik 3 Model")
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
    
    with st.expander("📌 Penjelasan Metrik"):
        st.markdown("""
        - **Accuracy**: Proporsi prediksi benar dari total prediksi
        - **F1-macro**: Rata-rata harmonik precision dan recall per kelas (tidak terbobot)
        - **Recall-macro**: Rata-rata recall per kelas (tidak terbobot)
        
        Recall-macro dipilih sebagai metrik utama karena dataset tidak seimbang.
        Model dengan recall tinggi berarti mampu mendeteksi serangan dengan baik.
        """)

# ---------------------------------------------------------------------
# TAB 4: CONFUSION MATRIX
# ---------------------------------------------------------------------
with tab_cm:
    st.subheader("📊 Confusion Matrix Per Model")
    
    pilihan_model = st.selectbox(
        "Pilih model untuk dilihat confusion matrix-nya", 
        list(hasil.keys()),
        index=list(hasil.keys()).index(model_terbaik)
    )
    data_model = hasil[pilihan_model]
    fig_cm, cm = fig_confusion(
        y_test, data_model['pred'], le.classes_,
        f"{pilihan_model} | Acc={data_model['accuracy']:.2%} | Recall={data_model['recall']:.2%}"
    )
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(fig_cm)
    with c2:
        st.subheader("📊 Recall per Kategori")
        per_class_recall = cm.diagonal() / cm.sum(axis=1)
        recall_df = pd.DataFrame({
            'Kategori': le.classes_,
            'Recall': per_class_recall
        }).sort_values('Recall', ascending=False)
        st.dataframe(recall_df.style.format({'Recall': '{:.2%}'}), 
                     use_container_width=True, hide_index=True)
        
        # Visualisasi recall
        fig_recall, ax = plt.subplots(figsize=(5, 3))
        colors_recall = ['#10B981' if r >= 0.8 else '#F59E0B' if r >= 0.5 else '#EF4444' 
                         for r in per_class_recall]
        ax.barh(le.classes_, per_class_recall, color=colors_recall)
        ax.set_title('Recall per Kategori')
        ax.set_xlabel('Recall')
        ax.set_xlim(0, 1)
        fig_recall.tight_layout()
        st.pyplot(fig_recall)
    
    st.caption(
        "📌 Kategori dengan recall paling rendah biasanya `U2R`/`R2L` karena "
        "jumlah sampelnya sangat sedikit di dataset."
    )

# ---------------------------------------------------------------------
# TAB 5: FEATURE IMPORTANCE
# ---------------------------------------------------------------------
with tab_fi:
    st.subheader("⭐ Feature Importance per Model")
    
    pilihan_model_fi = st.selectbox(
        "Pilih model untuk feature importance", 
        list(hasil.keys()),
        index=list(hasil.keys()).index(model_terbaik), 
        key="fi_select"
    )
    top_n = st.slider("Jumlah fitur teratas", 5, 25, 15)
    
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
            st.subheader("Top Fitur")
            st.dataframe(
                fi_df.reset_index(drop=True), 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "fitur": "Fitur",
                    "kepentingan": "Kepentingan"
                }
            )
    
    with st.expander("📌 Interpretasi Feature Importance"):
        st.markdown("""
        Feature importance menunjukkan seberapa besar kontribusi setiap fitur dalam pengambilan keputusan model.
        
        **Fitur penting yang umum ditemukan:**
        - `dst_host_same_srv_rate`: Tingkat keberhasilan layanan yang sama pada host tujuan
        - `dst_host_srv_count`: Jumlah koneksi ke layanan yang sama pada host tujuan
        - `count`: Jumlah koneksi ke host yang sama dalam 2 detik
        - `srv_count`: Jumlah koneksi ke layanan yang sama dalam 2 detik
        
        Fitur-fitur ini biasanya paling berpengaruh dalam membedakan serangan DoS dari lalu lintas normal.
        """)

# ---------------------------------------------------------------------
# TAB 6: COBA PREDIKSI
# ---------------------------------------------------------------------
with tab_predict:
    st.subheader("🔍 Uji Model dengan Data Uji")
    
    pilihan_model_pred = st.selectbox(
        "Pilih model untuk prediksi", 
        list(hasil.keys()),
        index=list(hasil.keys()).index(model_terbaik), 
        key="pred_select"
    )
    pipe_terpilih = hasil[pilihan_model_pred]['pipeline']

    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pilih Sampel")
        idx = st.number_input(
            "Indeks baris (0 s.d. {})".format(len(X_test) - 1),
            min_value=0, 
            max_value=len(X_test) - 1, 
            value=0, 
            step=1
        )
        
        sample = X_test.iloc[[idx]]
        
    with col2:
        st.subheader("Hasil Prediksi")
        
        actual_label = le.inverse_transform([y_test[idx]])[0]
        pred_label = le.inverse_transform(pipe_terpilih.predict(sample))[0]
        
        # Warna berdasarkan status
        status = "✅ Benar" if actual_label == pred_label else "❌ Salah"
        status_color = "green" if actual_label == pred_label else "red"
        
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa;'>
            <h4 style='color: #1f2937;'>🔹 Label Sebenarnya: <span style='color: #3B82F6;'>{actual_label}</span></h4>
            <h4 style='color: #1f2937;'>🔸 Prediksi Model: <span style='color: #3B82F6;'>{pred_label}</span></h4>
            <h3 style='color: {status_color};'>{status}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    with st.expander("Lihat Detail Fitur Sampel"):
        sample_df = sample.T.rename(columns={sample.index[0]: 'nilai'})
        sample_df['nilai'] = sample_df['nilai'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
        st.dataframe(sample_df, use_container_width=True, height=300)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p style='font-size: 14px;'>
            Dashboard Deteksi Intrusi Jaringan — NSL-KDD<br>
            Dibangun dengan Streamlit &amp; Scikit-learn
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================================
# DOWNLOAD MODEL TERLATIH (di sidebar)
# =====================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Unduh Model")
st.sidebar.caption(f"Model terbaik: {model_terbaik}")

buffer = io.BytesIO()
joblib.dump({'pipeline': hasil[model_terbaik]['pipeline'], 'label_encoder': le}, buffer)
st.sidebar.download_button(
    label=f"Unduh model_nsl.pkl",
    data=buffer.getvalue(),
    file_name="model_nsl.pkl",
    mime="application/octet-stream",
)

st.sidebar.markdown("---")
st.sidebar.caption("Versi 1.0 | © 2024")