import numpy as np
import pandas as pd
import streamlit as st
import bcrypt

# Configuração da página
st.set_page_config(
    page_title="Cálculo do diâmetro de Placas de Orifício",
    page_icon="logo.ico",  # Caminho do ícone
)

# Função para autenticação
def autenticar(username, password):
    if username in st.secrets.credentials:
        # Recuperar senha hash do arquivo secrets.toml
        hashed_pw = st.secrets.credentials[username]
        return bcrypt.checkpw(password.encode(), hashed_pw.encode())
    return False

# Função de logout
def sair():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.current_page = "login"
    st.rerun()  # Força a recarga da aplicação

# Tela de login
def login_page():
    st.image("logo.png", width=450)
    st.title("Login")
    st.write("Por favor, insira suas credenciais para acessar o sistema.")

    username = st.text_input("Usuário", key="login_username")
    password = st.text_input("Senha", type="password", key="login_password")

    if st.button("Entrar"):
        if autenticar(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.current_page = "main"
            st.rerun()  # Força a recarga da aplicação
        else:
            st.error("Usuário ou senha inválidos.")

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
            'kgf/cm²': 98.0665e3,
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
def carregar_planilhas():
    tubos = pd.read_excel("tubos.xlsx")
    coeficiente = pd.read_excel("coeficiente.xlsx")
    return tubos, coeficiente
 
# Inicialização do estado da sessão
if "resultados" not in st.session_state:
    st.session_state.resultados = None
if "calculo_feito" not in st.session_state:
    st.session_state.calculo_feito = False
 
# Carregar a planilha
planilha, coef_mat = carregar_planilhas()
 
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

def buscar_material(material):
    result = coef_mat.loc[coef_mat['material'] == material, 'coeficiente']
    return result.values[0] if not result.empty else None
    
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
 
def calculo_iterativo(v_normal,v_max,dp_max, p, qm, D, μ, estado, p1, k, tomada):
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
        if estado == "Liquido":
            β = calcular_beta(C_atual, ε, D, qm, dp_normal, p)
            C_atual = calcular_c(β, qm,p,D,μ, tomada)
        else:
            ε = calculo_epsilon(β, p1, p2, k)
            β = calcular_beta(C_atual, ε, D, qm, dp_normal, p)
            C_atual = calcular_c(β, qm,p,D,μ, tomada)

    d = β*D
    q = calculate_qm(C_atual, ε, β, D, dp_normal, p)
    Δω = calculate_delta_omega(β, C_atual, dp_max)
    v = calcular_velocidade(qm/p,D) # adicionar o caso com q (FS)
    Re_D = calcular_reynolds(p,v,D,μ) # segundo caso com essa nova velocidade

    return β, C_atual, d, q, Δω, v, Re_D, dp_normal

def recalcular_beta(tag, v_normal, v_max, dp_max, p, qm, D, μ, estado, p1, k, tomada, beta_initial, beta_condition, dp_adjustment, dp_processo, D_i, temperatura, alpha,densidade):
    """
    Recalcula valores para diferentes valores de Delta P até que a condição do Beta seja satisfeita.

    Args:
        tag (str): Tag do instrumento.
        v_normal (float): Vazão mássica normal.
        v_max (float): Vazão mássica máxima.
        dp_max (float): Pressão diferencial máxima inicial.
        p (float): Densidade.
        qm (float): Vazão mássica.
        D (float): Diâmetro interno da tubulação.
        μ (float): Viscosidade.
        estado (str): Estado do fluido ("Gas" ou "Liquido").
        p1 (float): Pressão de entrada.
        k (float): Fator de compressibilidade.
        tomada (str): Tipo de tomada ("Flange", "D", "Canto").
        beta_initial (float): Valor inicial do Beta.
        beta_condition (callable): Condição para continuar iterando.
        dp_adjustment (callable): Função para ajustar dp_max em cada iteração.

    Returns:
        List[Dict]: Lista de dicionários com os resultados recalculados.
    """
    beta_values = []
    dp_max_i = dp_max
    β = beta_initial

    while beta_condition(β, dp_max_i):
        dp_max_i = dp_adjustment(dp_max_i)  # Ajusta dp_max

        try:
            β, C_atual, d, q, Δω, v, Re_D, dp_normal = calculo_iterativo(v_normal, v_max, dp_max_i, p, qm, D, μ, estado, p1, k, tomada)
            d = d * np.sqrt(1 - (2*(alpha) * (temperatura-20)))
            beta = d/D_i # a condição de 20ºC
            # Armazena os valores em uma lista para exibição posterior
            beta_values.append({
                "Tag": tag,
                "Pressão diferencial máxima [mmH2O]": f"{dp_max_i / 9.80638:.2f}",
                "Pressão diferencial na vazão normal [mmH2O]": f"{dp_normal / 9.80638:.2f}",
                "Vazão mássica calculada [kg/s]": f"{q:.2f}",
                "Vazão mássica informada [kg/s]": f"{v_normal:.2f}",
                "Beta a temperatura operacional": f"{β:.5f}",
                "Beta @20ºC": f"{beta:.5f}",
                "Diâmetro da placa [mm] @20ºC": f"{d * 1000:.5f}",
                "Coeficiente C": f"{C_atual:.3f}",
                "Perda de carga permanente [bar]": f"{Δω* 1e-5:.5f}",
                "Velocidade [m/s]": f"{v:.2f}",
                "Número de Reynolds": f"{Re_D:.2f}",
                "Perda de carga máxima informada [bar]": f"{dp_processo:.2f}",
            })

        except Exception as e:
            st.error(f"Erro durante o cálculo iterativo: {e}")
            break

    return beta_values

def input_with_unit(label, unit_options, default_unit, key):
    st.write(label)  # Exibe o texto explicativo no topo
    col1, col2 = st.columns([3, 1])  # Define duas colunas para o valor e a unidade
    format_string = f"%.4f"
    with col1:
        input_value = st.number_input("Valor:", key=f"{key}_value", label_visibility="collapsed", format=format_string)
    with col2:
        unit = st.selectbox("Unidade:", unit_options, index=unit_options.index(default_unit), key=f"{key}_unit", label_visibility="collapsed")
    return input_value, unit

def exibir_sliders(β, Re_D, tomada, D, Δω, dp_processo):
    # Condições para validação
    if tomada in ["D", "Canto"]:
        lower_beta_limit = 0.1
        upper_beta_limit = 0.56
        valid_reynolds = (Re_D >= 5000 and lower_beta_limit <= β <= upper_beta_limit) or (Re_D >= (16000 * β**2))
    else:  # Tomada tipo "Flange"
        lower_beta_limit = 0.56
        upper_beta_limit = 1.0  # Limite superior arbitrário, já que β > 0.56
        valid_reynolds = Re_D >= (170 * (β**2) * D) and Re_D > 5000
    
    # Exibir o slider para β
    st.slider(
        "Beta Calculado",
        min_value=0.0,
        max_value=1.0,
        value=β,  # Mostrar o valor de β calculado
        step=0.01,
        format="%.3f",
        disabled=True,  # Apenas exibição
        key=f"beta_slider_{β:.5f}",
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
        key=f"reynolds_slider_{β:.5f}",
    )

    if β > 0.7 or β < 0.2:
        valid_beta = False
    else:
        valid_beta = True
    # Exibir mensagem de validação
    if valid_reynolds and valid_beta:
        st.success("Os valores calculados atendem aos critérios.")
    else:
        st.error("Os valores calculados NÃO atendem aos critérios!")

    # Exibe o slider com Δω e dp_processo como limite máximo
    st.slider(
        "Perda de Carga [bar]",
        min_value=0.0,
        max_value=dp_processo,
        value=Δω,
        step=0.01,
        format="%.5f",
        disabled=True,  # Apenas exibição
        key=f"perda_de_carga_slider_{β:.5f}",
    )

    # Exibir aviso se Δω for maior que dp_processo
    if Δω * 1e-5 > dp_processo:
        st.warning("A perda de carga está acima do definido.")

# Estilos personalizados para reduzir o espaçamento entre elementos
# Função auxiliar para entrada com unidade ao lado, com 'key' para cada elemento
# Função auxiliar para entrada com unidade ao lado dentro de um contêiner
# Configuração da interface Streamlit
# Entradas com chaves únicas

# Tela principal
def main_page():
    st.sidebar.write(f"Usuário logado: {st.session_state.username}")
    if st.sidebar.button("Sair"):
        sair()

    st.sidebar.image("logo.png", width=450)

    # Criar colunas para layout horizontal
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
            <h1 style="color: #E55204; margin: 0;">Cálculo do diâmetro de Placas de Orifício</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    tag = st.text_input("Informe o tag do instrumento (TagNo):", key="tag")

    # caso vapor com densidade em kg/m^3, vazão em kg/h - padrão atual do aeng
    # caso gas com vazão em Nm^3/h
    st.caption("Preencher o peso molecular apenas se as vazões estejam em Nm³/h e neste caso escolha Gas")
    estado_fluido = st.selectbox("Estado do fluido (FluidState):", ["Gas", "Liquido", "Vapor"], key="estado_fluido")

    tomada = st.selectbox("Informe o tipo de medição (ssdTipoTomada):", ["Flange", "D", "Canto"], key="tomada")

    delta_p_valor, delta_p_unidade = input_with_unit("Delta P na vazão máxima de cálculo (ssdDpCondicaoVazaoCalculo):",
                                                    ["Pa", "kPa", "MPa", "bar", "psi", "mmH2O", "inH2O"],
                                                    "Pa", key="delta_p")

    dp_processo = st.number_input("DP máximo permitido por processos (DPressureMax) [bar]:", key="dp_processo", format="%.4f")

    vazao_max_valor, vazao_max_unidade = input_with_unit("Vazão máxima de cálculo (ssdVazaoCalculo):",
                                                        ["kg/s", "g/s", "kg/h", "g/h", "m³/s", "m³/h"
                                                        ,"Nm³/h"], "kg/s", key="vazao_max")

    vazao_normal_valor, vazao_normal_unidade = input_with_unit("Vazão normal (FlowNormal):",
                                                            ["kg/s", "g/s", "kg/h", "g/h", "m³/s", "m³/h",
                                                            "Nm³/h"], "kg/s", key="vazao_normal")

    densidade_valor, densidade_unidade = input_with_unit("Densidade (Density):",
                                                        ["kg/m³", "g/cm³", "lb/ft³"],
                                                        "kg/m³", key="densidade")

    viscosidade_valor, viscosidade_unidade = input_with_unit("Viscosidade (Viscosity):",
                                                            ["Pa*s", "cP", "mPa*s", "P"],
                                                            "Pa*s", key="viscosidade")

    fator_compressibilidade = st.number_input("Fator de Compressibilidade Cp/Cv (CpCv):", key="fator_compressibilidade")

    peso_molecular = st.number_input("Peso molecular [g/mol]:", key="peso_molecular", format="%.4f")

    pressao_entrada_valor, pressao_entrada_unidade = input_with_unit("Pressão de entrada (PressureNormal):",
                                                                    ["Pa", "kPa", "MPa", "bar", "psi", "mmH2O", "inH2O","kgf/cm²"],
                                                                    "Pa", key="pressao_entrada")
    
    temperatura = st.number_input("Temperatura de Operação (TemperatureNormal) [ºC]:", key="temperatura", format="%.4f")

    busca_tipo = st.radio("Será informado o scheduleou a outra categoria da espessura da parede?", ["SCH", "Categoria"], key="busca_tipo")
    
    schedule = st.text_input("Schedule ou categoria (PipeSchedule):", key="schedule")

    diametro_linha = st.number_input("Diâmetro da linha (pdLineNominalDiam) [em polegadas]:", key="diametro_linha", format="%.4f")

    material_linha = st.selectbox("Material da linha:", ["Aço Carbono", "Aço Inox 304", "Aço Inox 310", "Aço Inox 316", "Plastico reforçado com fibra de vidro", "PTFE"], key="material_linha")

    material_placa = st.selectbox("Material da placa:", ["Aço Carbono", "Aço Inox 304", "Aço Inox 310", "Aço Inox 316", "Plastico reforçado com fibra de vidro", "PTFE"], key="eaterial_placa")

    if st.button("Calcular"):
        st.session_state.calculo_feito = True
        try:
            alpha = buscar_material(material_placa) # usado no calculo da placa (material placa)
            alpha = alpha * 1e-6

            alpha_m = buscar_material(material_linha) # usado no calculo da parade externa (material linha)
            alpha_m = alpha_m * 1e-6
            
            if busca_tipo == 'SCH':
                externo, parede = buscar_valores(diametro_linha, sch=float(schedule))
            elif busca_tipo == 'Categoria':
                externo, parede = buscar_valores(diametro_linha, denominacao=schedule)
            else:
                print("Opção inválida.")
                externo, parede = None, None
            
            if estado_fluido == "Gas":
                vazao_max_valor = 0.044*vazao_max_valor*peso_molecular
                vazao_normal_valor = 0.044*vazao_normal_valor*peso_molecular
                vazao_max_unidade = "kg/h"
                vazao_normal_unidade = "kg/h"
            else:
                vazao_max_valor = vazao_max_valor
                vazao_normal_valor = vazao_normal_valor
                vazao_max_unidade = vazao_max_unidade
                vazao_normal_unidade = vazao_normal_unidade
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
            D_i = (externo - 2*parede)/1000 # diâmetro interno m
            D = D_i * np.sqrt(1 + (2*(alpha_m) * (temperatura-20)))
            μ = viscosidade # viscosidade Pa.s
            estado = estado_fluido
            p1 = p1 # Pressão de entrada em Pa
            k = fator_compressibilidade # CpCv
            tomada = tomada
            
            β, C_atual, d, q, Δω, v, Re_D, dp_normal = calculo_iterativo(v_normal,v_max,dp_max, p, qm, D, μ, estado, p1, k, tomada)

            d = d * np.sqrt(1 - (2*(alpha) * (temperatura-20)))
            beta = d/D_i # a condição de 20ºC
            # Salvar resultados na sessão
            st.session_state.resultados = {
                "Parâmetro": [
                    "Tag",
                    "Pressão diferencial na vazão normal [mmH2O]",
                    "Vazão mássica calculada [kg/s]",
                    "Vazão massica informada [kg/s]",
                    "Beta a temperatura operacional",
                    "Beta @20ºC",
                    "Diâmetro da placa @20ºC [mm]",
                    "Coeficiente C",
                    "Perda de carga permanente [bar]",
                    "Velocidade [m/s]",
                    "Número de Reynolds",
                    "Perda de carga máxima informada [bar]",
                    "Diâmetro da placa externo a temperatura 20ºC[mm]",
                ],
                "Valor": [
                    tag,
                    f"{dp_normal / 9.80638:.2f}",
                    f"{q:.2f}",
                    f"{v_normal: .2f}",
                    f"{β:.5f}",
                    f"{beta:.5f}",
                    f"{d * 1000:.5f}",
                    f"{C_atual:.3f}",
                    f"{Δω * 1e-5:.2f}",
                    f"{v:.2f}",
                    f"{Re_D:.2f}",
                    f"{dp_processo:.2f}",
                    f"{D_i * 1000:.2f}",
                ],
            }
            st.success("Cálculo concluído!")

            if st.session_state.get("calculo_feito", False) and "resultados" in st.session_state:
                st.subheader("Resultados Calculados")
                resultados_df = pd.DataFrame(st.session_state.resultados)
                st.table(resultados_df)
        
                exibir_sliders(β, Re_D, tomada, D, Δω* 1e-5, dp_processo)

                if β > 0.7 or β < 0.2:
                    st.error("Beta inicial inválido. Recalculando com outros valores de Delta P...")

                    # Define condições específicas para cada caso
                    if β > 0.7:
                        beta_condition = lambda β, dp_max_i: β > 0.7 and dp_max_i < 50000
                        dp_adjustment = lambda dp_max_i: dp_max_i + 250 * 9.80638
                    elif β < 0.2:
                        beta_condition = lambda β, dp_max_i: β < 0.2 and dp_max_i > 0
                        dp_adjustment = lambda dp_max_i: dp_max_i - 250 * 9.80638

                    # Recalcula valores
                    beta_values = recalcular_beta(tag, v_normal, v_max, dp_max, p, qm, D, μ, estado, p1, k, tomada, β, beta_condition, dp_adjustment, dp_processo, D_i, temperatura, alpha, densidade)

                    # Exibe os resultados recalculados
                    if beta_values:
                        st.subheader("Resultados Recalculados para Diferentes Valores de ΔP")
                        recalculated_df = pd.DataFrame(beta_values).transpose() 
                        st.table(recalculated_df)

                    # Exibir sliders ajustados para os novos valores
                    novo_delta_omega = float(recalculated_df.loc["Perda de carga permanente [bar]"].iloc[-1])
                    novo_beta = float(recalculated_df.loc["Beta @20ºC"].iloc[-1])  # Último valor de Beta recalculado
                    novo_reynolds = float(recalculated_df.loc["Número de Reynolds"].iloc[-1])  # Último valor de Reynolds recalculado
                    exibir_sliders(novo_beta, novo_reynolds, tomada, D, novo_delta_omega, dp_processo)

                else:
                    valid_beta = True

        except Exception as e:
            st.error(f"Ocorreu um erro nos cálculos: {e}")

# Inicialização do estado da sessão
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"

# Redirecionar para a página atual
if st.session_state.current_page == "main" and st.session_state.logged_in:
    main_page()
else:
    login_page()
