import marimo

__generated_with = "0.13.10"
app = marimo.App()


@app.cell
def _():
    import random
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from datasets import load_dataset, Dataset

    # Load a model
    #model = SentenceTransformer('../training/models/german-nq-granite-embedding-107m-multilingual-exclude-pooling-prompts/checkpoint-4560')
    #model = SentenceTransformer("ibm-granite/granite-embedding-107m-multilingual")

    #model = SentenceTransformer("../training/models/german-nq-granite-embedding-278m-multilingual/checkpoint-4560")
    model = SentenceTransformer("ibm-granite/granite-embedding-278m-multilingual")
    #model = SentenceTransformer("../training/models/german-nq-paraphrase-multilingual-mpnet-base-v2/checkpoint-4560")
    return Dataset, load_dataset


@app.cell
def _(Dataset, load_dataset):
    revosax = load_dataset("csv", data_files="./data/training/training-data.csv", split="train")
    revosax = revosax.rename_column("result", "query").rename_column("chunk", "answer")
    revosax = revosax.select_columns(['query', 'answer'])
    revosax = revosax.train_test_split(test_size=0.2, seed=12)

    train_dataset: Dataset = revosax["train"]
    eval_dataset: Dataset = revosax["test"]

    print("obtained split:")
    print(f"training {train_dataset.shape}")
    print(f"eval     {eval_dataset.shape}")
    return eval_dataset, train_dataset


@app.cell
def _(eval_dataset, train_dataset):
    #test = eval_dataset.map(lambda x: x["query"])

    queries = {str(i): q for i, q in enumerate(eval_dataset["query"])}
    corpus  = {str(i): a for i, a in enumerate(eval_dataset["answer"])}
    corpus |= {str(i): a for i, a in enumerate(train_dataset["answer"][:5000],len(eval_dataset))} # plus 5000 random answers from the training set

    relevant_docs = {qid: {qid} for qid in queries.keys()}
    return


app._unparsable_cell(
    r"""
    # Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
    def compute_ir(queries,corpus,relevant_docs,model,name=\"foobar\")
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=name,
        )
        value = ir_evaluator(model)
        return value

    results = compute_ir(queries, corpus, relevant_docs, model, name=\"revosax-test-eval\")
    print(ir_evaluator.primary_metric)
    # => \"BeIR-touche2020-test_cosine_map@100\"
    print(results[ir_evaluator.primary_metric])
    # => 0.29335196224364596
    """,
    name="_"
)


@app.cell
def _(results):
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## IBM Granite 278M Multilingual Trained
    ```
    {'revosax-test-eval_cosine_accuracy@1': 0.6516696825135713,
     'revosax-test-eval_cosine_accuracy@3': 0.8803257114657016,
     'revosax-test-eval_cosine_accuracy@5': 0.9356802105609475,
     'revosax-test-eval_cosine_accuracy@10': 0.9753248889620003,
     'revosax-test-eval_cosine_precision@1': 0.6516696825135713,
     'revosax-test-eval_cosine_precision@3': 0.2934419038219005,
     'revosax-test-eval_cosine_precision@5': 0.18713604211218954,
     'revosax-test-eval_cosine_precision@10': 0.09753248889620005,
     'revosax-test-eval_cosine_recall@1': 0.6516696825135713,
     'revosax-test-eval_cosine_recall@3': 0.8803257114657016,
     'revosax-test-eval_cosine_recall@5': 0.9356802105609475,
     'revosax-test-eval_cosine_recall@10': 0.9753248889620003,
     'revosax-test-eval_cosine_ndcg@10': 0.8235574387806855,
     'revosax-test-eval_cosine_mrr@10': 0.7736648349639784,
     'revosax-test-eval_cosine_map@100': 0.7749149190019986}
     ```


    ## IBM Granite 107M Multilingual Trained

    ```
    {'revosax-test-eval_cosine_accuracy@1': 0.6322585951636782,
     'revosax-test-eval_cosine_accuracy@3': 0.8679059055765751,
     'revosax-test-eval_cosine_accuracy@5': 0.9213686461589077,
     'revosax-test-eval_cosine_accuracy@10': 0.9643033393650271,
     'revosax-test-eval_cosine_precision@1': 0.6322585951636782,
     'revosax-test-eval_cosine_precision@3': 0.289301968525525,
     'revosax-test-eval_cosine_precision@5': 0.1842737292317816,
     'revosax-test-eval_cosine_precision@10': 0.09643033393650272,
     'revosax-test-eval_cosine_recall@1': 0.6322585951636782,
     'revosax-test-eval_cosine_recall@3': 0.8679059055765751,
     'revosax-test-eval_cosine_recall@5': 0.9213686461589077,
     'revosax-test-eval_cosine_recall@10': 0.9643033393650271,
     'revosax-test-eval_cosine_ndcg@10': 0.8081156070335637,
     'revosax-test-eval_cosine_mrr@10': 0.7568607905957787,
     'revosax-test-eval_cosine_map@100': 0.7586377475723721}
    ```
    ## paraphrase-multilingual-mpnet
    ```
    {'revosax-test-eval_cosine_accuracy@1': 0.5957394308274387,
     'revosax-test-eval_cosine_accuracy@3': 0.8218456983056424,
     'revosax-test-eval_cosine_accuracy@5': 0.8822174699786149,
     'revosax-test-eval_cosine_accuracy@10': 0.9325546964961342,
     'revosax-test-eval_cosine_precision@1': 0.5957394308274387,
     'revosax-test-eval_cosine_precision@3': 0.27394856610188073,
     'revosax-test-eval_cosine_precision@5': 0.17644349399572296,
     'revosax-test-eval_cosine_precision@10': 0.09325546964961344,
     'revosax-test-eval_cosine_recall@1': 0.5957394308274387,
     'revosax-test-eval_cosine_recall@3': 0.8218456983056424,
     'revosax-test-eval_cosine_recall@5': 0.8822174699786149,
     'revosax-test-eval_cosine_recall@10': 0.9325546964961342,
     'revosax-test-eval_cosine_ndcg@10': 0.7712046375055522,
     'revosax-test-eval_cosine_mrr@10': 0.7186072531770789,
     'revosax-test-eval_cosine_map@100': 0.7214377020377937}
     ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
