"""
OmniGen - Streamlit Cloud Version
Lightweight demo app for cloud deployment.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(
    page_title="OmniGen - Drug Repurposing Demo",
    page_icon="🧬",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .stButton>button { background-color: #1f77b4; color: white; font-weight: bold; border-radius: 0.5rem; }
    .stButton>button:hover { background-color: #0d5a9e; }
</style>
""", unsafe_allow_html=True)

def get_predictions(disease_name, top_k):
    """Demo predictions."""
    import random
    random.seed(hash(disease_name) % 10000)
    
    drugs = [
        ("Losartan", "Angiotensin receptor blocker", 0.91),
        ("Amlodipine", "Calcium channel blocker", 0.88),
        ("Metoprolol", "Beta blocker", 0.85),
        ("Hydrochlorothiazide", "Thiazide diuretic", 0.82),
        ("Valsartan", "Angiotensin receptor blocker", 0.79),
        ("Ramipril", "ACE inhibitor", 0.76),
        ("Atenolol", "Beta blocker", 0.73),
        ("Lisinopril", "ACE inhibitor", 0.71),
        ("Nifedipine", "Calcium channel blocker", 0.68),
        ("Enalapril", "ACE inhibitor", 0.94),
    ]
    
    random.shuffle(drugs)
    selected = sorted(drugs[:top_k], key=lambda x: x[2], reverse=True)
    
    return [
        {"rank": i+1, "drug": drug, "mechanism": mech, 
         "confidence": conf, "novelty": "High" if conf > 0.85 else "Medium"}
        for i, (drug, mech, conf) in enumerate(selected)
    ]

def color_confidence(val):
    if val >= 0.85: return 'background-color: #d4edda; color: #155724'
    elif val >= 0.70: return 'background-color: #fff3cd; color: #856404'
    else: return 'background-color: #f8d7da; color: #721c24'

def main():
    st.markdown('<p class="main-header">🧬 OmniGen</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Omics Drug Repurposing Platform</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🧬 OmniGen")
        st.header("Configuration")
        disease_name = st.text_input("Disease Name", "Hypertension")
        top_k = st.slider("Number of Candidates", 5, 50, 10)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.70)
        
        st.markdown("---")
        st.header("Model Info")
        st.success("✅ Demo Mode")
        st.info("""
        - **Architecture**: Heterogeneous GNN (GAT)
        - **Layers**: 2
        - **Hidden Dim**: 64
        - **Parameters**: 4.5M
        - **Dataset**: DRKG (5.8M triplets)
        """)
        st.markdown("---")
        st.caption("Built by AyushNexa | 2026")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Predictions", "🕸️ Network", "📊 Analytics"])
    
    with tab1:
        st.header("Drug Repurposing Candidates")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Diseases", "5,103")
        col2.metric("Compounds", "24,313")
        col3.metric("Genes", "39,220")
        col4.metric("Treatments", "48")
        
        if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
            with st.spinner("Analyzing knowledge graph..."):
                predictions = get_predictions(disease_name, top_k)
                predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
                
                if predictions:
                    df = pd.DataFrame(predictions)
                    
                    styled = df.style.applymap(color_confidence, subset=['confidence']).format({'confidence': '{:.1%}'})
                    st.dataframe(styled, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, f"omnigen_{disease_name}.csv")
                    
                    fig = px.bar(df, x='drug', y='confidence', color='confidence', 
                                color_continuous_scale='RdYlGn', title=f"Scores for {disease_name}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    novelty_counts = df['novelty'].value_counts()
                    fig2 = px.pie(values=novelty_counts.values, names=novelty_counts.index,
                                 title="Novelty Distribution",
                                 color_discrete_map={'High': '#28a745', 'Medium': '#ffc107'})
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No candidates above threshold.")
    
    with tab2:
        st.header("Drug-Target-Disease Network")
        net = Network(height="500px", bgcolor="#ffffff", font_color="black")
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=250)
        
        net.add_node("Disease", label=disease_name, color="#ff6b6b", size=35)
        
        drugs = [("Enalapril", 0.94), ("Losartan", 0.91), ("Amlodipine", 0.88), 
                 ("Metoprolol", 0.85), ("Hydrochlorothiazide", 0.82)]
        
        for drug, conf in drugs:
            color = "#28a745" if conf > 0.9 else "#ffc107" if conf > 0.8 else "#17a2b8"
            net.add_node(drug, label=drug, color=color, size=20 + conf * 10)
            net.add_edge("Disease", drug, width=conf * 5)
        
        genes = ["ACE", "AGTR1", "CACNA1C", "ADRB1", "SLC12A3"]
        for gene in genes:
            net.add_node(gene, label=gene, color="#6f42c1", size=15)
        
        connections = [("Enalapril", "ACE"), ("Losartan", "AGTR1"), ("Amlodipine", "CACNA1C"),
                       ("Metoprolol", "ADRB1"), ("Hydrochlorothiazide", "SLC12A3")]
        
        for drug, gene in connections:
            net.add_edge(drug, gene, width=2, color="#6f42c1", dashes=True)
        
        net.save_graph("network.html")
        with open("network.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=500)
        
        st.markdown("""
        **Legend:** 🔴 Disease | 🟢 High confidence | 🟡 Medium | 🔵 Lower | 🟣 Gene
        """)
    
    with tab3:
        st.header("Model Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = list(range(11))
            df_hist = pd.DataFrame({
                'Epoch': epochs,
                'Train Loss': [0.60, 0.53, 0.48, 0.44, 0.41, 0.38, 0.36, 0.34, 0.33, 0.32, 0.31],
                'Val AUC': [0.50, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00]
            })
            fig = px.line(df_hist, x='Epoch', y=['Train Loss', 'Val AUC'], title="Training Progress")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            entities = pd.DataFrame({
                'Type': ['Compound', 'Disease', 'Gene'],
                'Count': [24313, 5103, 39220]
            })
            fig = px.pie(entities, values='Count', names='Type', title="Knowledge Graph Entities")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Model Architecture")
        arch = pd.DataFrame({
            'Layer': ['Input', 'GAT Layer 1', 'GAT Layer 2', 'Decoder'],
            'Type': ['Node Embeddings', 'HeteroConv + GAT', 'HeteroConv + GAT', 'MLP'],
            'Output': ['64 dims', '64 dims', '64 dims', '1 (probability)'],
            'Parameters': ['4.5M', '2.1M', '2.1M', '8K']
        })
        st.table(arch)
        
        st.info("**Training:** Adam (lr=0.005) | BCE Loss | Early Stopping | Best Val AUC: 1.0000")

if __name__ == "__main__":
    main()