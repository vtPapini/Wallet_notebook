
import streamlit as st
import json
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.express as px
import tempfile

#Funcs
def df_wallet_generate(stocks_code):
  YF_datas = pd.DataFrame()
  for stock in stocks_code:
    YF_datas[stock] = yf.download(stock, start='2021-01-01',progress=0,)['Close']
    YF_datas.dropna(inplace=True)
  return YF_datas

def wallet_data(yfDataFrame,Ativos=list,Pesos=list,Fundo=float):
  df = pd.DataFrame()
  df['Ativo'],df['Peso']= Ativos,Pesos
  df.set_index('Ativo',inplace=True)
  df['Fundo_Inicial(R$)'] = df['Peso']*(Fundo/100)
  
  taxa_crescimento = yfDataFrame.loc[Data_inic:].copy()

  df['Retorno_(%)'] = round( ( (taxa_crescimento.iloc[-1] / taxa_crescimento.iloc[1]) -1 ) * 100 , 2)
  
  df['Fundo_Atual(R$)'] = round(df['Fundo_Inicial(R$)'] * ( (df['Retorno_(%)']/100)+1),2)
  
  #-Adicionando a soma
  df_sum = df.copy()
  df_sum.loc['Σ']= df_sum.sum().round(2)
  
  return df, df_sum

def pie_chart(data, peso_col):
    figure = px.pie(data, values=data[peso_col], names=data.index, title='Distribuição da Carteira', width=900, height=400)
    figure.update_traces(textposition='outside', textinfo='percent+label')
    return figure

def history(data):
  figure = px.line(title = 'Histórico de fechamento das ações (Início 2021-01-01)',width=1200, height=800)
  for i in data.columns:
    figure.add_scatter(x = data.index, y = data[i], name = i)
  figure.update_xaxes(
                      dtick="M1",
                      tickformat="%b\n%Y",
                      rangeslider_visible=True)
  return figure

def grow_tx(df,date_init):
  taxa_crescimento = df.loc[date_init:].copy()
  for i in taxa_crescimento:
    taxa_crescimento[i] = (((taxa_crescimento[i] / taxa_crescimento[i][0])) -1) * 100

  figure = px.line(taxa_crescimento,
                   title = f'Taxa de crescimento das ações em relação ao início {date_init}',
                   width=1200, height=800)
  
  figure.update_layout(yaxis_title='Crescimento (%)',
                       xaxis_title='Data',
                       legend=dict(title='Ações'))
  
  figure.update_xaxes(
                      dtick="M1",
                      tickformat="%b\n%Y")

  return taxa_crescimento, figure

def box_var(df,date_init):
  variacao_d = df.loc[date_init:].copy()

  for i in variacao_d:
    variacao_d[i] = ((variacao_d[i] / variacao_d[i].shift(1)) - 1)*100 
  
  figure = px.box(variacao_d,
                  title = f'Boxplot variações diárias',
                  orientation='h',
                  #color_discrete_sequence=["purple"],
                  width=800, height=800)
  
  figure.update_layout(yaxis_title='Ação',
                       xaxis_title='Variações diária (%)',
                       legend=dict(title='Ações'))
  return figure

def var_d_log(df,date_init):
  kk = df.loc[date_init:].copy()
  kk = np.log(kk)
  for i in kk:
    kk[i] = ( (kk[i] / kk[i].shift(1)) - 1)

  figure = px.line(kk,
                  title = f'Variação logarítmica diária da carteira (Início {date_init})',
                  width=1200, height=800)
  figure.update_layout(yaxis_title='Variação (logarítmica)',
                       xaxis_title='Período',
                       legend=dict(title='Ações'))
  figure.update_xaxes(
                      dtick="M1",
                      tickformat="%b\n%Y",
                      rangeslider_visible=True)  
  
  return figure

def freq_var_acum(df,date_init):
  kk = df.loc[date_init:].copy()
  for i in kk:
    kk[i] = ((kk[i] / kk[i].shift(1))-1) * 100

  figure = px.histogram(kk,facet_col_wrap=2,
                  title = f'Distribuição total da variação diária (Início {date_init})',
                  width=1200, height=800)
  
  figure.update_layout(yaxis_title='Frequência',
                       xaxis_title='Variação diária (%)',
                       legend=dict(title='Ações'))

  return figure

def freq_var(df,date_init):

  #---------Gréfico de freq das variações percentuais **INDIVIDUAIS**

  # configurando dataset
  kk = df.loc[date_init:].copy()
  for i in kk:
    kk[i] = ( (kk[i] / kk[i].shift(1)) - 1) * 100
  kk.dropna(inplace=True)
  
  
  # Criando figura vazia com dois subplots
    # Ordenando número de colunas
  num_rows = round(len(kk.columns)/4)
  if len(kk.columns) % 4 == 1:
    num_rows = round(len(kk.columns)/4) + 1

  fig = sp.make_subplots(rows=num_rows, cols=4, subplot_titles=kk.columns)

  # Criando histogramas individuais
  col_ = 1
  row_ = 1
  for i in kk.columns:
    
    fig.add_histogram(x=kk[i], name=i, row=row_, col=col_)
    col_ += 1

    if col_ > 4:
      col_ = 1
      row_ += 1

  # Atualizar o layout da figura
  fig.update_layout(title=f'Distribuição da variação percentual diária (Início {date_init})',
                    legend=dict(title='Ações'),
                    width=1200, height=800)
  
  return fig


def freq_var2(df,date_init):
  
  #---------Gréfico de freq das variações percentuais **SOBREPOSTAS**
  # configurando dataset
  kk = df.loc[date_init:].copy()
  for i in kk:
    kk[i] = ( (kk[i] / kk[i].shift(1)) - 1) * 100
  kk.dropna(inplace=True)
  
  # Criando figura vazia com subplots
  fig = go.Figure()
  for i in kk.columns:    
    fig.add_trace(go.Histogram(x=kk[i],name=i))
  # Atualizar o layout da figura
  fig.update_layout(title=f'Distribuição da variação percentual diária (Início {date_init})',
                    legend=dict(title='Ações'),
                    yaxis_title='Frequência',
                    xaxis_title='Variação diária (%)',
                    barmode='overlay',
                    width=1200, height=800)
  
  return fig

def correlation_pct(df: pd.DataFrame):
  corr = df.pct_change().corr()
  mask = np.triu(np.ones_like(corr, dtype=bool))
  heat = go.Heatmap(
                    z=corr.mask(mask),
                    x=corr.columns,
                    y=corr.columns,
                    hoverinfo="z", #Shows hoverinfo for null values
                    colorscale=[[0,'#4682B4'],[0.5, '#FFFFFF'], [1, '#FF4500']],
                    zmin=-1,
                    zmax=1,
                    xgap = 4, # Sets the horizontal gap (in pixels) between bricks
                    ygap = 4
                  )

  layout = go.Layout(
                     title_text='Correlação variativa', 
                     title_x=0.5, 
                     width=600, 
                     height=600,
                     xaxis_showgrid=False,
                     yaxis_showgrid=False,
                     yaxis_autorange='reversed',
                     template='plotly_white'
                    )

  fig = go.Figure(data=[heat],layout=layout)

  return fig

#----------------------------------Funcs do streamlit

def genarate_JSON():
    info = {"Data_inic":f"{Data_inic}","Ativos":ativos,"Pesos":pesos,"Fundo":fundo}
    obj = json.dumps(info,indent=2)
    return obj.encode('utf-8') # retorna o JSON em bytes

def save_JSON(json_bytes):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_file.write(json_bytes)
        return temp_file.name

def carrega_dados_st(ativos,pesos,fundo):
    ativos_list = ativos.split('/')
    pesos_list = pesos.split('/')
    pesos_list = [float(p) for p in pesos_list]
    yfdata = df_wallet_generate(ativos_list)
    df, df_sum = wallet_data(yfdata,ativos_list,pesos_list,fundo)
    
    st.success("Carteira carregada!", icon="✅")
    k1, k2 = st.columns([3, 1])
    k1.dataframe(df_sum,use_container_width=True)
    k2.metric(label="Retorno Atual ($)", value=df_sum["Fundo_Atual(R$)"].iloc[-1], delta= f'{df_sum["Retorno_(%)"].iloc[-1]}%')

    st.plotly_chart(pie_chart(df, peso_col="Peso"), theme="streamlit", use_container_width=True)
    return df, yfdata


#-----------------------------------------------------------------------

st.title('Wallet project - LongTime')

pad_fundo = 0
pad_pesos = ''
pad_ativo = ''
pad_data = None

st.subheader('Insira os dados da sua carteira 👇')

with st.expander("Adicionando Dados"):
  st.markdown('Selecione a data inicial da carteira')  
  Data_inic = st.date_input("Dados até **2021**.",value=pad_data)

  st.markdown('Selecione a nomeclatura dos ativos em carteira de acordo com o site: ➡ https://finance.yahoo.com')
  ativos = st.text_area('Insira os ativos separados por /:',value=pad_ativo)
  
  st.markdown('Atenção ❗ A somatória dos pesos **precisam** dar 100 (ex:50/50)')  
  pesos = st.text_input('Insira os pesos percentuais separados por /:',value=pad_pesos)
  
  fundo = st.number_input('Insira o fundo inicial:',value=pad_fundo)

col1, col2 = st.columns(2)
if col1.button('Carregar'):
    df, yfdata = carrega_dados_st(ativos,pesos,fundo)
col2.download_button("Download",data=genarate_JSON(),file_name="My_Wallet_project.JSON")


c1 = st.container()
c1.subheader('Já fez download da carteira aqui antes? \n Então faça o upload aqui 👇')

with c1.expander("UPLOAD CARTEIRA"):
   J_info = st.file_uploader("Arraste aqui",type='json')

if J_info is not None: 
    info = pd.read_json(J_info, orient='index').T
    
    Data_inic = info.Data_inic.values[0]
    df, yfdata = carrega_dados_st(info.Ativos.values[0],info.Pesos.values[0],info.Fundo.values[0])
    pad_fundo = info.Fundo.values[0]
    pad_pesos = info.Pesos.values[0]
    pad_ativo = info.Ativos.values[0]
    pad_data = pd.to_datetime(info.Data_inic.values[0])


selectbox = st.radio('Relatórios',('*Esconder*','Desempenho','Risco'),horizontal=True)
Relat = st.empty()

if selectbox == '*Esconder*':
   Relat.empty()

elif selectbox == 'Desempenho':
   
   tab1, tab2 = Relat.tabs(["Histórico", "Crescimento(%)"])
   
   tab1.plotly_chart(history(yfdata))
   _, fig = grow_tx(yfdata,Data_inic)
   tab2.plotly_chart(fig)
else:
   tab1, tab2, tab3, tab4, tab5 = Relat.tabs(["Correlação","BoxPlots", "Volatividade","Freq. Var","Freq. Acumulada"])
   
   tab1.plotly_chart(correlation_pct(yfdata))

   tab2.plotly_chart(box_var(yfdata,Data_inic))
   
   tab3.plotly_chart(var_d_log(yfdata,Data_inic))

   tab4.plotly_chart(freq_var2(yfdata,Data_inic))
   
   tab4.plotly_chart(freq_var(yfdata,Data_inic))
   
   tab5.plotly_chart(freq_var_acum(yfdata,Data_inic)) 