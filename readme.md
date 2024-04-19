# Projeto de Classificação de Mamografia

Este projeto utiliza uma rede neural simples, especificamente um Perceptron Multicamadas (MLP), para classificar resultados de mamografias.

## Estrutura do Projeto

O projeto consiste nos seguintes arquivos:

* [`mamografia_multiple_tests.py`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "mamografia_multiple_tests.py"): Este script realiza vários testes com diferentes configurações de unidades ocultas e funções de ativação para encontrar o melhor modelo.
* [`mamografia_test.py`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "mamografia_test.py"): Este script cria um modelo MLP com 5 unidades de entrada, 40 unidades ocultas e 1 unidade de saída. Ele treina o modelo, faz previsões no conjunto de testes e calcula a acurácia do modelo.
* [`MLP.py`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "MLP.py"): Este arquivo contém a implementação da classe [`MLP_Classifier`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "MLP.py"), que representa o modelo de rede neural.
* [`dadosmamografia.xlsx`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "dadosmamografia.xlsx"): Este é o conjunto de dados usado para treinar e testar o modelo.
* [`pseudocode.txt`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "pseudocode.txt"): Este arquivo contém pseudocódigo para o projeto.

## Como usar

Para usar este projeto, você precisa ter Python e as bibliotecas numpy, matplotlib e pandas instaladas.

Para executar um dos scripts, use o seguinte comando no terminal:

`python mamografia_test.py`

ou

`python mamografia_multiple_tests.py`

## Detalhes da Implementação

O modelo MLP é implementado na classe [`MLP_Classifier`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "MLP.py") no arquivo [`MLP.py`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "MLP.py"). Esta classe tem métodos para inicializar o modelo, obter os pesos, realizar forward e backpropagation.

Os scripts [`mamografia_test.py`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "mamografia_test.py") e [`mamografia_multiple_tests.py`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "mamografia_multiple_tests.py") usam a classe [`MLP_Classifier`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "MLP.py") para criar um modelo, treiná-lo com os dados de [`dadosmamografia.xlsx`](vscode-file://vscode-app/c:/Users/claud/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "dadosmamografia.xlsx"), fazer previsões e calcular a acurácia do modelo.
