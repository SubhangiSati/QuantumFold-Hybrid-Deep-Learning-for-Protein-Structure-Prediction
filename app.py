import streamlit as st
from stmol import showmol
import py3Dmol
from protein_model import get_protein_structure  # Import the function

# st.set_page_config(layout = 'wide')
st.sidebar.title('Protein Structure Prediction using Quantum')
st.sidebar.write('This tool predicts the protein structure based on quantum-inspired methods leveraging advanced computational models.')

# stmol visualization function
def render_mol(pdb):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb, 'pdb')
    pdbview.setStyle({'cartoon': {'color': 'spectrum'}})
    pdbview.setBackgroundColor('white')
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True)
    showmol(pdbview, height=500, width=800)

# Protein sequence input
DEFAULT_SEQ = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
txt = st.sidebar.text_area('Input sequence', DEFAULT_SEQ, height=275)

# Protein structure prediction
def update(sequence=txt):
    # Call the API to get the predicted structure
    pdb_string = get_protein_structure(sequence)

    # Save the structure to a file
    with open('predicted.pdb', 'w') as f:
        f.write(pdb_string)

    # Display protein structure
    st.subheader('Visualization of Predicted Protein Structure')
    render_mol(pdb_string)

    # Add a download button
    st.download_button(
        label="Download PDB File",
        data=pdb_string,
        file_name='predicted.pdb',
        mime='text/plain',
    )

# Sidebar button
predict = st.sidebar.button('Predict', on_click=update)

if not predict:
    st.warning('👈 Enter protein sequence data!')
