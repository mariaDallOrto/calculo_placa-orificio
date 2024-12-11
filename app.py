import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
def converter_unidade(valor, unidade):
    fatores_de_conversao = {
        'vazao': {
            'kg/s': 1,  # Unidade de saída já é kg/s
            'g/s': 1e-3,
            'kg/h': 1 / 3600,
            'g/h': 1 / 3.6e6,
            'm³/s': 1000,  # Converter m³/s para kg/s usando densidade
            'm³/h': 1000 / 3600,
        },
        'pressao': {
            'Pa': 1,  # Unidade de saída já é Pa
            'kPa': 1e3,
            'MPa': 1e6,
            'bar': 1e5,
            'psi': 6894.76,
            'mmH2O': 9.80665,
            'inH2O': 249.08891,
        },
        'densidade': {
            'kg/m³': 1,  # Unidade de saída já é kg/m³
            'g/cm³': 1e3,
            'lb/ft³': 16.0185,
        },
        'diametro': {
            'm': 1,  # Unidade de saída já é m
            'cm': 1e-2,
            'mm': 1e-3,
            'in': 0.0254,
        },
        'viscosidade': {
            'Pa*s': 1,  # Unidade de saída já é Pa*s
            'cP': 1e-3,
            'mPa*s': 1e-3,
            'P': 0.1,
        },
    }
 
    for grandeza, unidades in fatores_de_conversao.items():
        if unidade in unidades:
            fator = unidades[unidade]
            return valor * fator, grandeza
 
    raise ValueError(f"Unidade '{unidade}' não suportada.")
 
def converter_vazao_volumetrica_para_massica(valor, unidade, densidade):
    # Converte vazão volumétrica para vazão massiva usando a densidade
    if unidade in ['m³/s', 'm³/h']:
        valor_kg_h = valor * densidade if unidade == 'm³/h' else valor * densidade * 3600
        return valor_kg_h / 3600, "kg/s" # Retorna em kg/s
    return valor, unidade  # Caso já seja massiva, retorna o valor original
 
# Cache para carregar a planilha
@st.cache_data
def carregar_planilha():
    return pd.read_excel("tubos.xlsx")
 
# Inicialização do estado da sessão
if "resultados" not in st.session_state:
    st.session_state.resultados = None
if "calculo_feito" not in st.session_state:
    st.session_state.calculo_feito = False
 
# Carregar a planilha
planilha = carregar_planilha()
 
# Função para buscar valores
def buscar_valores(d, sch=None, denominacao=None):
    if sch is not None:
        # Caso o usuário forneça 'sch'
        resultado = planilha[(planilha['Ø Nominal'] == d) & (planilha['Schedule'] == sch)]
    elif denominacao is not None:
        # Caso o usuário forneça a 'Denominação'
        resultado = planilha[(planilha['Ø Nominal'] == d) & (planilha['Denominação'] == denominacao)]
    else:
        # Caso nenhuma entrada adicional seja fornecida
        return None, None
 
    # Verificar se encontrou resultados
    if not resultado.empty:
        # Retornar as colunas desejadas
        externo = resultado['Ø Externo (mm)'].values[0]
        parede = resultado['Parede (mm)'].values[0]
        return externo, parede
    else:
        return None, None  # Se não encontrar resultados
 
def calcular_beta(C, ε, D, qm, dp_normal, p):
    numerator = (C * ε * np.pi * D**2)**2
    denominator = 8 * qm**2
    inner_term = numerator / denominator
 
    # Calcular beta usando a fórmula
    β = 1 / ((1 + (inner_term * dp_normal * p))**(1/4))
    return β
 
def calcular_velocidade(qm,D):
    Q = qm
    A = np.pi * (D**2) / 4  # Área da seção transversal
    v = Q / A  # Velocidade média
    return v
 
def calcular_reynolds(p,v,D,μ):
    Re = (p * v * D) / μ
    return Re
 
def calcular_c(β, qm, p, D, μ, tomada):
    # Componentes intermediários
    v = calcular_velocidade(qm/p,D)
    Re_D = calcular_reynolds(p,v,D,μ)
 
    if tomada == "Flange":
        L1 = 25.4/(D*1000) # depende do tipo de medição
        L2 = L1
    elif tomada == "D":
        L1 = 1
        L2 = 0.47
    elif tomada == "Canto":
        L1 = 0
        L2 = 0
    M2 = 2*L2/(1-β)
 
    A = (19000*β/Re_D)*0.8
 
    termo1 = 0.5961 + 0.0261 * β**2 - 0.216 * β**8 + 0.000521 * ((1e6*β / Re_D)**0.7)
    termo2 = (0.0188 + 0.0063 * A) * (β**3.5) * ((1e6 / Re_D)**0.3)
    termo3 = (0.043 + 0.080 * np.exp(-10 * L1) - 0.123 * np.exp(-7 * L1)) * (1 - 0.11 * A)
    termo4 = (β**4 / (1 - β**4))
    termo5 = -0.031 * (M2 - 0.8 * (M2**1.1)) * (β**1.3)
    termo6 = 0.011*(0.75 - β)*(2.8 - (D/25.4))
 
    # Calcular o valor de C
    if D*1000 < 71.12:
        C = termo1 + termo2 + termo3 * termo4 + termo5 + termo6
    else: C = termo1 + termo2 + termo3 * termo4 + termo5
   
    return C
 
def calculate_qm(C, ε, β, D, dp_normal, p):
    # Componentes intermediários
    term_1 = 1 / np.sqrt((1 / β**4) - 1)
    term_2 = (np.pi / 4) * D**2
    term_3 = np.sqrt(2 * dp_normal * p)
   
    # Calcular qm
    qm = C * ε * term_1 * term_2 * term_3
    return qm
 
def calculo_epsilon(β, p1, p2, k):
    factor = 0.351 + (0.256 * (β**4)) + (0.93 * (β**8))
    pressure_term = 1 - (p2 / p1)**(1 / k)
   
    epsilon = 1 - factor * pressure_term
    return epsilon
 
def calculate_delta_omega(β, C, dp):
    """
    Calcula Δω conforme a equação fornecida.
 
    Parâmetros:
    - beta: Relação de diâmetros (adimensional)
    - C: Coeficiente de descarga (adimensional)
    - Δp: Diferença de pressão (Pa)
 
    Retorna:
    - Δω (adimensional)
    """
    numerator = np.sqrt(1 - (β**4) * (1 - C**2)) - C * (β**2)
    denominator = np.sqrt(1 - (β**4) * (1 - C**2)) + C * (β**2)
   
    Δω = (numerator / denominator) * dp
    return Δω
 
# Configuração da interface Streamlit
st.title("Cálculo de Vazão e Parâmetros em Tubulações")
# Estilos personalizados para reduzir o espaçamento entre elementos
# Função auxiliar para entrada com unidade ao lado, com 'key' para cada elemento
# Função auxiliar para entrada com unidade ao lado dentro de um contêiner
def input_with_unit(label, value, unit_options, default_unit, key):
    st.write(label)  # Exibe o texto explicativo no topo
    col1, col2 = st.columns([3, 1])  # Define duas colunas para o valor e a unidade
    with col1:
        input_value = st.number_input("Valor:", value=value, key=f"{key}_value", label_visibility="collapsed")
    with col2:
        unit = st.selectbox("Unidade:", unit_options, index=unit_options.index(default_unit), key=f"{key}_unit", label_visibility="collapsed")
    return input_value, unit
 
# Entradas com chaves únicas
estado_fluido = st.selectbox("Estado do fluido:", ["Gas", "Liquido"], key="estado_fluido")
 
delta_p_valor, delta_p_unidade = input_with_unit("Delta P na vazão máxima de cálculo:", 2500.0,
                                                 ["Pa", "kPa", "MPa", "bar", "psi", "mmH2O", "inH2O"],
                                                 "Pa", key="delta_p")
 
vazao_max_valor, vazao_max_unidade = input_with_unit("Vazão máxima de cálculo:", 35.00000,
                                                     ["kg/s", "g/s", "kg/h", "g/h", "m³/s", "m³/h"],
                                                     "kg/s", key="vazao_max")
 
vazao_normal_valor, vazao_normal_unidade = input_with_unit("Vazão normal:", 27.30000,
                                                           ["kg/s", "g/s", "kg/h", "g/h", "m³/s", "m³/h"],
                                                           "kg/s", key="vazao_normal")
 
densidade_valor, densidade_unidade = input_with_unit("Densidade:", 26.0045,
                                                     ["kg/m³", "g/cm³", "lb/ft³"],
                                                     "kg/m³", key="densidade")
 
viscosidade_valor, viscosidade_unidade = input_with_unit("Viscosidade:", 0.03,
                                                         ["Pa*s", "cP", "mPa*s", "P"],
                                                         "Pa*s", key="viscosidade")
 
fator_compressibilidade = st.number_input("Fator de Compressibilidade Cp/Cv:", value=1.4, key="fator_compressibilidade")
 
busca_tipo = st.radio("Será informado o scheduleou a outra categoria da espessura da parede?", ["SCH", "Categoria"], key="busca_tipo")
schedule = st.text_input("Schedule ou categoria:", value="", key="schedule")
 
diametro_linha = st.number_input("Diâmetro da linha (em polegadas):", value=3.00, key="diametro_linha")
 
pressao_entrada_valor, pressao_entrada_unidade = input_with_unit("Pressão de entrada:", 90.8000,
                                                                 ["Pa", "kPa", "MPa", "bar", "psi", "mmH2O", "inH2O"],
                                                                 "Pa", key="pressao_entrada")
 
tomada = st.selectbox("Informe o tipo de medição:", ["Flange", "D", "Canto"], key="tomada")

tag = st.text_input("Informe o tag do instrumento:", key="tag")
 
if st.button("Calcular"):
    st.session_state.calculo_feito = True
    try:
        if busca_tipo == 'SCH':
            externo, parede = buscar_valores(diametro_linha, sch=float(schedule))
        elif busca_tipo == 'Categoria':
            externo, parede = buscar_valores(diametro_linha, denominacao=schedule)
        else:
            print("Opção inválida.")
            externo, parede = None, None
 
        # Converter valores para unidades padrão
        dp_max, _ = converter_unidade(delta_p_valor, delta_p_unidade)
        densidade, _ = converter_unidade(densidade_valor, densidade_unidade)
       
        # Verifica se as vazões são volumétricas e as converte para massivas
        vazao_max_valor, vazao_max_unidade = converter_vazao_volumetrica_para_massica(vazao_max_valor, vazao_max_unidade, densidade)
        v_max, _ = converter_unidade(vazao_max_valor, vazao_max_unidade)
       
        vazao_normal_valor, vazao_normal_unidade = converter_vazao_volumetrica_para_massica(vazao_normal_valor, vazao_normal_unidade, densidade)
        v_normal, _ = converter_unidade(vazao_normal_valor, vazao_normal_unidade)
 
        viscosidade, _ = converter_unidade(viscosidade_valor, viscosidade_unidade)
        p1, _ = converter_unidade(pressao_entrada_valor, pressao_entrada_unidade)
 
        v_normal = v_normal # vazão normal kg/s
        v_max = v_max # vazão máxima kg/s
        dp_max = dp_max # delta p máximo Pa
        p =  densidade # densidade Pa
        qm = v_normal # kg/s
        externo = externo # mm
        parede = parede # mm
        D = (externo - 2*parede)/1000 # diâmetro interno m
        μ = viscosidade # viscosidade Pa.s
        estado = estado_fluido
        p1 = p1 # Pressão de entrada em Pa
        k = fator_compressibilidade # CpCv
        tomada = tomada
 
        dp_normal = dp_max*((v_normal/v_max)**2)
        p2 = p1 - dp_normal
 
        C = 0.6 # quando for líquido altera só esse
        ε = 1 # caso de gás compressível altera esse também
 
        β = calcular_beta(C, ε, D, qm, dp_normal, p)
        C_atual = calcular_c(β, qm,p,D,μ, tomada)
        C_anterior = C
        max_iterations = 1000
        iterations = 0
        while(abs(C_atual - C_anterior) > 10**-6):
            iterations += 1
            if iterations > max_iterations:        
                st.error("O cálculo não convergiu após 1000 iterações.")
                break
            C_anterior = C_atual
            if estado == "Gas":
                ε = calculo_epsilon(β, p1, p2, k)
            β = calcular_beta(C_atual, ε, D, qm, dp_normal, p)
            C_atual = calcular_c(β, qm,p,D,μ, tomada)
 
        d = β*D
        q = calculate_qm(C_atual, ε, β, D, dp_normal, p)
        q = q*3600/1000
        Δω = calculate_delta_omega(β, C_atual, dp_normal)
        v = calcular_velocidade(qm/p,D) # adicionar o caso com q (FS)
        Re_D = calcular_reynolds(p,v,D,μ) # segundo caso com essa nova velocidade
 
 
        # Salvar resultados na sessão
        st.session_state.resultados = {
            "Parâmetro": [
                "Tag",
                "Pressão diferencial normal [mmH2O]",
                "Vazão calculada [m³/h]",
                "Beta",
                "Diâmetro da placa [mm]",
                "Coeficiente C",
                "Perda de carga permanente [Pa]",
                "Velocidade [m/s]",
                "Número de Reynolds",
                "vazão massica informada [kg/s]",
            ],
            "Valor": [
                tag,
                f"{dp_normal / 9.80638:.2f}",
                f"{q:.2f}",
                f"{β:.3f}",
                f"{d * 1000:.2f}",
                f"{C_atual:.3f}",
                f"{Δω:.2f}",
                f"{v:.2f}",
                f"{Re_D:.2f}",
                f"{v_normal: .2f}"
            ],
        }
        st.success("Cálculo concluído!")
 
        if st.session_state.get("calculo_feito", False) and "resultados" in st.session_state:
            st.subheader("Resultados Calculados")
            resultados_df = pd.DataFrame(st.session_state.resultados)
            st.table(resultados_df)
            #Limites para o Beta
            # Criar o gráfico
            # Configuração do valor de beta, Reynolds e os critérios
       
            # Condições para validação
            if tomada in ["D", "Canto"]:
                lower_beta_limit = 0.1
                upper_beta_limit = 0.56
                valid_reynolds = Re_D >= 5000 and lower_beta_limit <= β <= upper_beta_limit
            else:  # Tomada tipo "Flange"
                lower_beta_limit = 0.56
                upper_beta_limit = 1.0  # Limite superior arbitrário, já que β > 0.56
                valid_reynolds = Re_D >= (16000 * β**2) and β > lower_beta_limit
           
            # Estilo personalizado para o slider
            slider_color = "green" if valid_reynolds else "red"
            st.markdown(
                f"""
            <style>
                .streamlit-slider {{
                    background: {slider_color} !important;
                }}
            </style>
                """,
                unsafe_allow_html=True,
            )
           
            # Exibir o slider para β
            st.slider(
                "Beta Calculado",
                min_value=0.0,
                max_value=1.0,
                value=β,  # Mostrar o valor de β calculado
                step=0.01,
                format="%.3f",
                disabled=True,  # Apenas exibição
                key="beta_slider",
            )
           
            # Exibir o slider para Reynolds
            reynolds_lower_limit = 5000 if tomada in ["D", "Canto"] else 16000 * β**2
            st.slider(
                "Número de Reynolds",
                min_value=0.0,
                max_value=float(max(int(Re_D * 1.5), int(reynolds_lower_limit * 1.5))),
                value=Re_D,  # Mostrar o valor de Reynolds calculado
                step=100.0,
                disabled=True,  # Apenas exibição
                key="reynolds_slider",
            )
           
            # Exibir mensagem de validação
            if valid_reynolds :
                st.success("Os valores calculados atendem aos critérios.")
            else:
                st.error("Os valores calculados NÃO atendem aos critérios!")
 
    except Exception as e:
        st.error(f"Ocorreu um erro nos cálculos: {e}")
