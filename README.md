# calculo_placa-orificio
Cálculo do diâmetro de uma placa de orifício com base nas variáveis de processo e na ISO 5167.

O programa foi desenvolvido com base na solução iterativa descrita pelo cálculo realiado dentro do software AVEVA e com base nas fórmulas presentes na norma ISO 5167.

## Variáveis de entrada

Devem ser especificados os seguintes valores e unidades:
- Estado do Fluido
- Delta P Máximo
- Vazão Máxima
- Vazão Normal
- Densidade 
- Viscosidade 
- Fator de compressibilidade
- Schedule
- Diâmetro externo da Linha
- Pressão de Entrada
- Tipo de tomada

## Fórmulas utilizadas

As fórmulas são todas baseadas na taxa de fluxo mássico. É realizado um procedimento iterativo ao calcular 𝛽. Após cada iteração, 𝐶 é corrigido (e, para fluidos compressíveis, 𝜖) e recalculado.

### Cálculo de Beta
![formula-beta](https://github.com/user-attachments/assets/c3ab8c40-b503-4c3b-9f60-9e9a5723201e)

### Cálculo do coeficiente de descarga
![formula-c](https://github.com/user-attachments/assets/d0225870-d5d2-422f-a394-3caef791c860)

Existem algumas ressalvas durante este cálculo.
- Quando D < 71,12mm, deve ser adicionado o termo seguinte a equação:
  
![termo-c](https://github.com/user-attachments/assets/fed5490f-89c6-4779-8335-ca90bc180b7d)

- Dependendo do tipo de tomada, os valores de L1 e L2 são modificados.
- 
### Cálculo do fator de expansão

