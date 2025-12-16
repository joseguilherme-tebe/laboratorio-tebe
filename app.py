import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import hilbert, butter, filtfilt

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(layout="wide", page_title="Tebe Sensores - Treinamento de Vibração")

# Estilo CSS (Mantido Original)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #e6f3ff; border-bottom: 2px solid #0068c9; }
    .js-plotly-plot .plotly .modebar { orientation: v; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Tebe Sensores | Laboratório de Análise de Vibração")
st.markdown("**Simulador de Fenômenos Físicos e Processamento de Sinal**")

tab1, tab2, tab3, tab4 = st.tabs([
    "Módulo 1: Fundamentos & Integração", 
    "Módulo 2: Digitalização & Resolução", 
    "Módulo 3: Harmônicos & Modulação",
    "Módulo 4: Envelope & Rolamentos"
])

layout_default = dict(margin=dict(l=20, r=20, t=40, b=20))

# ==============================================================================
# MÓDULO 1: FUNDAMENTOS
# ==============================================================================
with tab1:
    st.header("Módulo 1: A Física da Vibração e Integração")
    st.markdown("Entenda como a frequência afeta a amplitude nas diferentes grandezas.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parâmetros da Fonte")
        f1 = st.slider("Frequência (Hz)", 1, 200, 60, key="m1_f")
        amp_g = st.slider("Amplitude Aceleração (g - Pico)", 0.1, 10.0, 1.0, key="m1_a")
        
        omega = 2 * np.pi * f1
        amp_vel = (amp_g * 9806.65) / omega
        amp_disp = (amp_g * 9806.65 * 1000) / (omega**2)
        
        st.info(f"""
        **Valores Convertidos (Pico):**
        - Aceleração: **{amp_g:.2f} g**
        - Velocidade: **{amp_vel:.2f} mm/s**
        - Deslocamento: **{amp_disp:.2f} µm**
        """)
        
        st.markdown("---")
        st.subheader("Configuração do Espectro")
        unidade_fft = st.radio("Unidade de Amplitude (FFT)", ["Pico (Pk)", "RMS", "Pico-a-Pico (Pk-Pk)"])
        viz_mode = st.radio("Modo de Visualização", ["Formas de Onda (Tempo)", "Espectros (FFT)"])

    with col2:
        fs = 2000
        duration = 1.0
        t = np.linspace(0, duration, int(fs*duration))
        
        y_acc = amp_g * np.sin(omega * t)
        y_vel = amp_vel * np.sin(omega * t - np.pi/2)
        y_disp = amp_disp * np.sin(omega * t - np.pi)
        
        if viz_mode == "Formas de Onda (Tempo)":
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                subplot_titles=("Aceleração (g)", "Velocidade (mm/s)", "Deslocamento (µm)"),
                                vertical_spacing=0.1)
            
            rms_acc = np.sqrt(np.mean(y_acc**2))
            rms_vel = np.sqrt(np.mean(y_vel**2))
            rms_disp = np.sqrt(np.mean(y_disp**2))
            
            fig.add_trace(go.Scatter(x=t[:200], y=y_acc[:200], name="Acel", line=dict(color='blue')), row=1, col=1)
            fig.add_hline(y=rms_acc, line_dash="dash", line_color="orange", annotation_text="RMS", row=1, col=1)
            
            fig.add_trace(go.Scatter(x=t[:200], y=y_vel[:200], name="Vel", line=dict(color='green')), row=2, col=1)
            fig.add_hline(y=rms_vel, line_dash="dash", line_color="orange", annotation_text="RMS", row=2, col=1)
            
            fig.add_trace(go.Scatter(x=t[:200], y=y_disp[:200], name="Desl", line=dict(color='#FFCC00')), row=3, col=1)
            fig.add_hline(y=rms_disp, line_dash="dash", line_color="orange", annotation_text="RMS", row=3, col=1)
            
            fig.update_layout(height=600, title="Integração do Sinal (Linha Laranja = Nível RMS)", **layout_default)
            st.plotly_chart(fig, use_container_width=True)
            
        else: # FFT
            n = len(t)
            freqs = np.fft.rfftfreq(n, d=1/fs)
            
            def aplicar_escala(fft_array):
                if unidade_fft == "RMS": return fft_array * 0.7071
                if unidade_fft == "Pico-a-Pico (Pk-Pk)": return fft_array * 2
                return fft_array 
            
            fft_acc = aplicar_escala(np.abs(np.fft.rfft(y_acc)) / n * 2)
            fft_vel = aplicar_escala(np.abs(np.fft.rfft(y_vel)) / n * 2)
            fft_disp = aplicar_escala(np.abs(np.fft.rfft(y_disp)) / n * 2)
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=(f"Espectro Aceleração ({unidade_fft})", f"Espectro Velocidade ({unidade_fft})", f"Espectro Deslocamento ({unidade_fft})"),
                                vertical_spacing=0.1)
            
            fig.add_trace(go.Scatter(x=freqs, y=fft_acc, name="Acel", mode='lines', fill='tozeroy', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=freqs, y=fft_vel, name="Vel", mode='lines', fill='tozeroy', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=freqs, y=fft_disp, name="Desl", mode='lines', fill='tozeroy', line=dict(color='#FFCC00')), row=3, col=1)
            
            fig.update_layout(height=600, title=f"Comparação Espectral - Escala: {unidade_fft}", xaxis3_range=[0, 200], **layout_default)
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# MÓDULO 2: DIGITALIZAÇÃO
# ==============================================================================
with tab2:
    st.header("Módulo 2: Configuração do Coletor")
    
    col_cfg, col_vis = st.columns([1, 2])
    
    with col_cfg:
        st.subheader("Parâmetros de Coleta")
        sig_freq = st.number_input("Freq. Sinal Real (Hz)", 10, 2000, 80)
        st.markdown("---")
        amostragem = st.selectbox("Taxa de Amostragem (Fs)", [128, 256, 512, 1024, 2048, 4096, 8192], index=4)
        num_linhas = st.selectbox("Número de Linhas (Resolução)", [400, 800, 1600, 3200, 6400, 12800], index=0)
        
        fmax = amostragem / 2.56
        res_freq = fmax / num_linhas
        tempo_coleta = num_linhas / fmax
        
        st.markdown("### Cálculos Matemáticos:")
        st.latex(r"F_{max} = \frac{F_s}{2.56} = " + f"{fmax:.1f} Hz")
        st.latex(r"Resolução = \frac{F_{max}}{Linhas} = " + f"{res_freq:.3f} Hz")
        st.latex(r"Tempo = \frac{Linhas}{F_{max}} = " + f"{tempo_coleta:.2f} s")
        
        if sig_freq > (amostragem / 2):
            st.error(f"🚨 ALIASING! Sinal ({sig_freq}Hz) > Nyquist ({amostragem/2}Hz)")

    with col_vis:
        fig_time = go.Figure(go.Bar(
            x=[tempo_coleta], y=["Tempo de Coleta (s)"], orientation='h', 
            text=[f"{tempo_coleta:.2f} s"], textposition='auto', marker_color='#0068c9'
        ))
        fig_time.update_layout(height=120, margin=dict(l=0, r=0, t=30, b=0), title="Duração da Coleta (Impacto das Linhas)")
        st.plotly_chart(fig_time, use_container_width=True)

        n_samples_coleta = int(tempo_coleta * amostragem)
        t_analog = np.linspace(0, tempo_coleta, 10000) 
        y_analog = np.sin(2 * np.pi * sig_freq * t_analog)
        t_digital = np.arange(n_samples_coleta) / amostragem
        y_digital = np.sin(2 * np.pi * sig_freq * t_digital)
        
        fig_dig = go.Figure()
        limit_viz = min(0.2, tempo_coleta)
        fig_dig.add_trace(go.Scatter(x=t_analog[t_analog<limit_viz], y=y_analog[t_analog<limit_viz], name='Sinal Real', line=dict(color='lightgray')))
        fig_dig.add_trace(go.Scatter(x=t_digital[t_digital<limit_viz], y=y_digital[t_digital<limit_viz], mode='markers+lines', name='Sinal Digital', line=dict(color='red', dash='dash')))
        fig_dig.update_layout(title="Digitalização no Tempo (Zoom)", height=300, xaxis_title="Tempo (s)")
        st.plotly_chart(fig_dig, use_container_width=True)
        
        # Correção FFT Resolução
        fft_vals = np.abs(np.fft.rfft(y_digital)) / n_samples_coleta * 2
        freqs = np.fft.rfftfreq(n_samples_coleta, d=1/amostragem)
        
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(x=freqs, y=fft_vals, mode='lines+markers', name='Espectro', fill='tozeroy', line=dict(color='blue'), marker=dict(size=4)))
        fig_spec.add_vline(x=fmax, line_dash="dash", annotation_text="Fmax")
        
        xmax = max(fmax*1.1, sig_freq*1.1)
        fig_spec.update_layout(title=f"Espectro Resultante (Resolução = {res_freq:.3f} Hz)", height=300, xaxis_range=[0, xmax], xaxis_title="Frequência (Hz)")
        st.plotly_chart(fig_spec, use_container_width=True)

# ==============================================================================
# MÓDULO 3: HARMÔNICOS E MODULAÇÃO
# ==============================================================================
with tab3:
    st.header("Módulo 3: Análise Espectral Avançada")
    st.markdown("Experimente separadamente os fenômenos de baixa e alta frequência.")
    
    st.subheader("A. Harmônicos (Baixa/Média Frequência)")
    col_harm_input, col_harm_plot = st.columns([1, 2])
    
    with col_harm_input:
        f_fund = st.number_input("Frequência Fundamental (1x)", 10.0, 200.0, 30.0)
        amp_fund = st.slider("Amplitude 1x", 0.0, 2.0, 1.0)
        
        st.markdown("**Adicionar Componentes:**")
        tipo_harm = st.selectbox("Tipo de Harmônico", ["Nenhum", "Harmônicos Inteiros (Série)", "Sub-Harmônico (0.5x)", "Inter-Harmônico (1.5x)"])
        
        ordem_h = 0
        if tipo_harm == "Harmônicos Inteiros (Série)":
            ordem_h = st.slider("Até qual ordem?", 2, 20, 3, help="Exibe 1x, 2x, 3x... até a ordem selecionada")
        elif tipo_harm == "Sub-Harmônico (0.5x)":
            ordem_h = 0.5
        elif tipo_harm == "Inter-Harmônico (1.5x)":
            ordem_h = 1.5
            
        amp_h = st.slider("Amplitude dos Harmônicos", 0.0, 2.0, 0.5)

    with col_harm_plot:
        t3 = np.linspace(0, 1.0, 4000)
        y_h = amp_fund * np.sin(2 * np.pi * f_fund * t3)
        
        if tipo_harm == "Harmônicos Inteiros (Série)":
            for i in range(2, int(ordem_h) + 1):
                y_h += amp_h * np.sin(2 * np.pi * (f_fund * i) * t3)
        elif tipo_harm != "Nenhum":
            y_h += amp_h * np.sin(2 * np.pi * (f_fund * ordem_h) * t3)
            
        fft_h = np.abs(np.fft.rfft(y_h)) / len(t3) * 2
        freqs_h = np.fft.rfftfreq(len(t3), d=1/4000)
        
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=freqs_h, y=fft_h, mode='lines', name='Espectro', line=dict(color='green')))
        max_ordem = ordem_h if tipo_harm != "Nenhum" else 1
        max_x_view = f_fund * (max_ordem + 2)
        fig_h.update_layout(title="Espectro de Harmônicos (Linha)", height=300, xaxis_range=[0, max_x_view], xaxis_title="Hz")
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    st.subheader("B. Modulação e Bandas Laterais (Alta Frequência)")
    
    col_mod_input, col_mod_plot = st.columns([1, 2])
    with col_mod_input:
        f_portadora = st.number_input("Frequência Portadora (Alta)", 100, 2000, 500)
        f_moduladora = st.number_input("Frequência Moduladora (Baixa)", 1, 100, 10)
        indice_mod = st.slider("Índice de Modulação (Amplitude)", 0.0, 2.0, 0.5)
        n_bandas = st.slider("Quantidade de Pares de Bandas", 1, 10, 1)
        
    with col_mod_plot:
        y_mod = 1.0 * np.sin(2 * np.pi * f_portadora * t3)
        for n in range(1, n_bandas + 1):
            amp_banda = indice_mod / (n + 1)
            y_mod += amp_banda * np.sin(2 * np.pi * (f_portadora - n * f_moduladora) * t3)
            y_mod += amp_banda * np.sin(2 * np.pi * (f_portadora + n * f_moduladora) * t3)
        
        fft_m = np.abs(np.fft.rfft(y_mod)) / len(t3) * 2
        freqs_m = np.fft.rfftfreq(len(t3), d=1/4000)
        
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=freqs_m, y=fft_m, mode='lines', name='Bandas Laterais', line=dict(color='orange')))
        zoom_range = [f_portadora - f_moduladora*(n_bandas+2), f_portadora + f_moduladora*(n_bandas+2)]
        fig_m.update_layout(title=f"Espectro com Bandas Laterais (Zoom em {f_portadora}Hz)", height=300, xaxis_range=zoom_range, xaxis_title="Hz")
        st.plotly_chart(fig_m, use_container_width=True)

# ==============================================================================
# MÓDULO 4: ENVELOPE E ROLAMENTOS
# ==============================================================================
with tab4:
    st.header("Módulo 4: Análise de Envelope (O Pulo do Gato)")
    st.markdown("Simulação de defeito em rolamento excitando frequência natural.")
    
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.subheader("Dados da Máquina")
        rpm_m4 = st.number_input("Rotação do Eixo (RPM)", 600, 3600, 1800, key="m4_rpm")
        bpfo_order = st.number_input("Ordem de Falha (BPFO)", 1.0, 10.0, 4.2)
        bpfo_hz = (rpm_m4 / 60) * bpfo_order
        st.info(f"Frequência de Falha Calculada: **{bpfo_hz:.2f} Hz**")
        
    with row1_c2:
        st.subheader("Física do Impacto")
        fn = st.slider("Frequência Natural (Igrejinha)", 500, 3000, 1200, step=100)
        amortecimento = st.slider("Amortecimento do Impacto", 10, 100, 30)
        st.caption("Alto Amortecimento = Lubrificação Boa (Sinal Baixo). Baixo Amortecimento = Lubrificação Ruim (Sinal Alto).")
        filtro_fc = st.number_input("Filtro Passa-Alta (Hz)", 0, 2000, 500)
    
    # CORREÇÃO FÍSICA: Amplitude do Impacto Inversa ao Amortecimento
    # Quanto maior o amortecimento, menor a amplitude inicial do impacto.
    fator_amortecimento = 80.0 / amortecimento # Fator de escala inverso
    
    fs_env = 8192
    duration_env = 1.0
    t_env = np.linspace(0, duration_env, int(fs_env * duration_env))
    
    trem = np.zeros_like(t_env)
    periodo_idx = int(fs_env / bpfo_hz)
    trem[::periodo_idx] = 1.0 * fator_amortecimento # APLICAÇÃO DO FATOR NA FORÇA
    
    resposta = np.exp(-amortecimento * t_env[:1000]) * np.sin(2 * np.pi * fn * t_env[:1000])
    resposta = np.pad(resposta, (0, len(t_env)-len(resposta)), 'constant')
    
    sinal_bruto = np.convolve(trem, resposta[:1000], mode='same')
    
    # Ruído Colorido para "Sujar" a Igrejinha
    ruido_branco = np.random.normal(0, 0.3, len(t_env))
    b_noise, a_noise = butter(2, [fn-400, fn+400], btype='bandpass', fs=fs_env)
    ruido_colorido = filtfilt(b_noise, a_noise, ruido_branco)
    
    sinal_bruto += ruido_colorido 
    sinal_bruto += 0.5 * np.sin(2 * np.pi * (rpm_m4/60) * t_env) 
    sinal_bruto += np.random.normal(0, 0.1, len(t_env)) 
    
    # Processamento
    b, a = butter(4, filtro_fc/(fs_env/2), btype='high')
    sinal_filtrado = filtfilt(b, a, sinal_bruto)
    sinal_analitico = hilbert(sinal_filtrado)
    envelope = np.abs(sinal_analitico)
    
    fft_bruto = np.abs(np.fft.rfft(sinal_bruto)) / len(t_env) * 2
    freqs_bruto = np.fft.rfftfreq(len(t_env), d=1/fs_env)
    fft_env = np.abs(np.fft.rfft(envelope - np.mean(envelope))) / len(t_env) * 2
    
    # Visualização
    st.markdown("### 1. O Problema (Espectro de Aceleração)")
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=freqs_bruto, y=fft_bruto, mode='lines', name='Espectro Padrão', line=dict(color='gray', width=1)))
    fig_raw.add_vrect(x0=fn-200, x1=fn+200, annotation_text="Igrejinha", fillcolor="orange", opacity=0.2)
    fig_raw.update_layout(title="Espectro Realista (Com Ruído de Banda Larga na Ressonância)", xaxis_range=[0, 2000], height=300, **layout_default)
    st.plotly_chart(fig_raw, use_container_width=True)
    
    st.markdown("### 2. A Solução (Espectro de Envelope)")
    col_proc1, col_proc2 = st.columns(2)
    
    with col_proc1:
        st.markdown("**Sinal no Tempo (Demodulado)**")
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t_env[:500], y=sinal_filtrado[:500], name="Filtrado", line=dict(color='lightgray')))
        fig_time.add_trace(go.Scatter(x=t_env[:500], y=envelope[:500], name="Envelope", line=dict(color='red', width=2)))
        fig_time.update_layout(height=300, title="Extraindo o perfil do impacto")
        st.plotly_chart(fig_time, use_container_width=True)
        
    with col_proc2:
        st.markdown("**FFT do Envelope**")
        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(x=freqs_bruto, y=fft_env, mode='lines', name='Envelope FFT', line=dict(color='red')))
        for i in range(1, 4):
            fig_env.add_vline(x=bpfo_hz*i, line_dash="dot", annotation_text=f"{i}x")
        fig_env.update_layout(height=300, xaxis_range=[0, 500], title=f"Falha detectada em {bpfo_hz:.1f} Hz")
        st.plotly_chart(fig_env, use_container_width=True)
