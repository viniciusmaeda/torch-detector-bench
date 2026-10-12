# torch-detector-bench

Treinamento e comparação de YOLO, RT-DETR, Faster R-CNN e DETR com **cross-validation
de cinco dobras sobre todas as imagens** de um dataset de detecção COCO. O projeto
exporta resultados reproduzíveis e artefatos para artigos científicos.

## Dataset e protocolo

```text
datasets/images/all/
├── imagem_001.jpg
├── ...
└── _annotations.coco.json
```

Configure `dataset.images` e `dataset.annotations` em [config/config.yaml](config/config.yaml).
As classes e os IDs são inferidos automaticamente do JSON, com um mapeamento global
fixo. Categorias sem nenhuma anotação, como `objects` neste dataset, são excluídas;
`__background__` é acrescentado apenas à interface interna. Os arquivos originais
não são alterados. As imagens são compartilhadas por links simbólicos e os rótulos
YOLO são convertidos uma vez, com o mesmo mapeamento em todas as dobras.

A divisão usa estratificação multirrótulo iterativa: classes raras são distribuídas
primeiro, considerando presença por imagem. A implementação adapta a alocação de
Sechidis et al. (2011), acrescentando limites de capacidade para equilibrar o número
de imagens. Não é uma promessa de balanceamento exato de todas as classes/objetos.
O manifesto registra a seed e a impressão SHA256 das imagens, anotações e protocolo.

Para o dataset de 1.562 imagens, a seed 42 produz:

| Dobra | Treinamento | Avaliação |
|---|---:|---:|
| 1 | 1.250 | 312 |
| 2 | 1.250 | 312 |
| 3 | 1.250 | 312 |
| 4 | 1.249 | 313 |
| 5 | 1.249 | 313 |

As 31 classes anotadas aparecem nas cinco dobras. A classe `jau` aparece em 3, 2, 2,
3 e 3 imagens, respectivamente. Cada imagem é avaliada exatamente uma vez e usada
para treinamento nas outras quatro execuções. Não há teste fixo separado.

Se houver frames do mesmo vídeo, variantes aumentadas ou outras imagens da mesma
origem, forneça `cross_validation.group_file`, um CSV com `image_id,group_id` e uma
linha por imagem. Os grupos ficam inteiros em uma dobra; as quantidades podem variar.
Nomes repetidos não identificam automaticamente a mesma origem. O modo padrão assume
imagens independentes. Baixe imagens sem augmentation offline e aplique augmentation
somente ao treinamento de cada dobra.

## Ambiente

```bash
conda env create -n torch-detector-bench -f environment/torch-detector-bench.yml
conda activate torch-detector-bench
```

O ambiente inclui PyTorch, Torchvision, Ultralytics, Transformers, pycocotools,
NumPy, pandas, SciPy e Matplotlib. CUDA é opcional. Os pesos pré-treinados são
carregados pelos respectivos frameworks; o primeiro treinamento pode baixá-los.
`--prepare-only` não baixa pesos e não importa os detectores.

## Execução

Validar os dados, gerar as dobras e os gráficos de distribuição:

```bash
python main.py --prepare-only
```

Treinar as cinco dobras de um modelo, ou todos os quatro sequencialmente:

```bash
python main.py --model yolo
python main.py --model rtdetr
python main.py --model fasterrcnn
python main.py --model detr
python main.py --model all
```

Execuções específicas e regeneração de relatórios:

```bash
python main.py --model detr --fold 1 2
python main.py --model detr
python main.py --reports-only
python main.py --model yolo --config config/config.yaml
```

Cada comando de treinamento inicia uma execução completa desde os pesos de origem:
as cinco dobras são treinadas novamente, mesmo se uma execução anterior foi interrompida
ou concluída. Não é necessário um parâmetro de retomada. `--fold` limita a nova execução
às dobras indicadas, para verificações específicas; não acumula dobras de execuções anteriores.
Os resultados anteriores dos modelos escolhidos e os relatórios são arquivados em
`archive/<data-hora>/`, preservando os registros para o artigo e evitando misturar históricos.
Com `--model all`, os quatro modelos anteriores são arquivados antes de qualquer treinamento.
Resultados de outros modelos só podem ser comparados se o código e o ambiente forem iguais.
Para alterar o dataset ou o protocolo das dobras, use um novo `experiment.name`.

Defina épocas, hiperparâmetros, limiares e métricas estatísticas **antes** da execução.
Cada dobra cria modelo, otimizador e scheduler novos. O checkpoint avaliado é
`last.pt`, da última época definida, sem early stopping pela dobra de avaliação.
Ultralytics fornece os pesos EMA da última época. Seu `best.pt` e sua validação final
nativa podem existir, mas não são usados no benchmark comum.

Os tamanhos configurados são aplicados a todos os detectores. Faster R-CNN recebe
imagens em [0,1], pois a normalização já ocorre dentro do Torchvision. O mAP comum
usa **pycocotools/COCO**, não o cálculo simplificado da versão anterior.

## Artefatos para o artigo

Tudo fica em `results/<experiment.name>/`:

```text
fish_cv5_seed42/
├── resolved_config.json
├── source/                       # snapshot do código identificado por SHA256
├── archive/<data-hora>/          # resultados e relatórios das execuções anteriores
├── splits/
│   ├── manifest.json
│   ├── audit.json
│   ├── image_folds.csv / .tex
│   ├── class_distribution.csv / .tex
│   └── fold_01/...fold_05/        # listas de imagens e data.yaml
├── data/images/ e data/labels/   # imagens por symlinks e rótulos compartilhados
├── <modelo>/
│   ├── run.json
│   └── fold_01/...fold_05/
│       ├── config.json e environment.json
│       ├── completed.json e metrics.json
│       ├── checkpoints/weights/last.pt
│       ├── training_logs/ e training_curves/
│       └── evaluation/          # predições, métricas, contagens, PR e confusão
└── reports/
    ├── methods_and_statistics.md
    ├── completion_status.csv / .tex
    ├── fold_metrics.csv / .tex
    ├── summary_statistics.csv / .tex
    ├── manuscript_table.csv / .tex
    ├── per_class_folds.csv / .tex
    ├── per_class_summary.csv / .tex
    ├── paired_comparisons.csv / .tex  # após concluir todos os modelos previstos
    ├── paired_manuscript_table.csv / .tex # comparação compacta da métrica primária
    ├── learning_curves.csv
    ├── figures/                 # PNG 300 dpi, PDF e SVG
    └── out_of_fold/<modelo>/    # análise descritiva após as cinco dobras
```

Os gráficos incluem distribuição de classes, suporte por dobra, boxplots com os
pontos reais de cada dobra e linhas pareadas, AP por classe, curvas de treinamento
com média ± desvio-padrão, curvas precisão–recall, matrizes de confusão, concordância
de contagens/Bland–Altman e exemplos qualitativos com ground truth e predições.
Resultados parciais são identificados como parciais e não entram nas comparações.

AP usa IoU 0,50:0,05:0,95, 101 pontos de recall, todos os tamanhos e no máximo 100
detecções por imagem. O score mínimo para AP é 0,001. Precision, Recall e F1 são
micro métricas no limiar fixo 0,25 e IoU 0,50; contagens usam o mesmo limiar.
MAPE exclui contagens verdadeiras iguais a zero e informa o suporte válido. AP de
classes ausentes, correlações indefinidas e R² indefinido permanecem NA.
O export comum não suporta anotações COCO `iscrowd`: ele recusa esses dados.

As métricas finais dos modelos usam o mesmo avaliador. Curvas nativas por época são
apenas monitoramento: Ultralytics usa critérios nativos diferentes dos adotados na
avaliação final. Não use suas métricas por época como resultado do benchmark comum.
FPS inclui preprocessamento, inferência e pós-processamento, exclui leitura em disco,
usa aquecimento e sincronização CUDA. Interprete a comparação junto com hardware e
batch size. `Parameters` conta todos os parâmetros, inclusive os congelados, antes
da fusão de BatchNorm na inferência. `Training_seconds` inclui monitoramento e
exportação dos artefatos da fase de treinamento.

## Estatística

As unidades das tabelas comparativas são as **cinco dobras**, nunca as épocas ou
bounding boxes. São exportados média, desvio-padrão amostral, mediana, mínimo,
máximo e quantidade de valores válidos. Os resultados OOF agrupados são descritivos
e ficam separados: AP agrupado não equivale à média dos APs das dobras.

O padrão define `mAP50_95` como métrica primária e compara `mAP50_95`, `mAP50`, `F1`,
`MAE` e `RMSE`. As comparações só começam quando os modelos de
`statistics.required_models` têm todas as dobras completas. O teste t pareado usa
a correção aproximada de Nadeau–Bengio:

```text
SE = sqrt((1/k + mean(n_avaliação/n_treinamento)) × variância amostral das diferenças)
df = k − 1
```

As saídas incluem diferença A−B, IC aproximado de 95%, p bilateral, efeito pareado
dz descritivo e p ajustado por Holm em toda a família de pares de modelos/métricas.
Os ICs das diferenças não têm ajuste simultâneo; o ajuste de Holm aplica-se aos p.
Diferenças não nulas com variância zero ficam com inferência indefinida.

Com cinco dobras de um único dataset, os testes são **exploratórios**, com baixo
poder e pressupostos difíceis de verificar. A correção considera aproximadamente
a dependência causada pelos conjuntos de treinamento sobrepostos. Não se afirma
superioridade científica automaticamente. Ajustar hiperparâmetros após observar
esses resultados introduz viés de seleção; seria necessária validação aninhada ou
dados externos para separar essa etapa de uma avaliação independente.

O arquivo `methods_and_statistics.md` registra o protocolo, definições e limitações
para auxiliar a redação da seção de métodos. Referências:

- [Sechidis et al. e estratificação multirrótulo](https://github.com/trent-b/iterative-stratification)
- [Avaliador COCO oficial](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- [Correção estatística e dependência entre dobras](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html)

## Verificação e R opcional

```bash
python -m unittest discover -s tests -v
Rscript evaluation/graficos.R results/fish_cv5_seed42/reports/fold_metrics.csv
```

O script R usa métricas finais por dobra; não transforma épocas correlacionadas em
amostras estatísticas. Requer ggplot2, dplyr e tidyr. Python já gera os relatórios
completos, sem depender de R.

Datasets, pesos e resultados são ignorados pelo Git. Os módulos Python dentro de
`datasets/` são versionáveis para preservar o código de preparação e carregamento.
