import mygene
import pandas as pd

import config


def map_gene_id():
    mg = mygene.MyGeneInfo()

    gene_info_table = pd.read_csv(config.gene_info_table)

    missing_cnt = 0
    total_cnt = 0
    expanded_gene_info_table = gene_info_table
    for idx, ensembl_id in enumerate(gene_info_table['ensembl_id']):
        total_cnt += 1
        try:
            result = mg.query(ensembl_id, scopes='ensembl.gene', fields='entrezgene,refseq,ensembl.gene')

            if not result['hits']:
                missing_cnt += 1
                print(f"❌ 未找到结果：{ensembl_id}")
                continue

            hit = result['hits'][0]

            entrez_id = hit.get('entrezgene', 'N/A')
            refseq_id = hit.get('refseq', {}).get('rna') if isinstance(hit.get('refseq'), dict) else hit.get('refseq',
                                                                                                             'N/A')

            if isinstance(refseq_id, list):
                refseq_id = refseq_id[0]
            if entrez_id is None or refseq_id is None:
                missing_cnt += 1
                print(f"❌ None：{ensembl_id}")

            if entrez_id is not None:
                expanded_gene_info_table.at[idx, 'entrez_id'] = entrez_id
            if refseq_id is not None:
                expanded_gene_info_table.at[idx, 'refseq_id'] = refseq_id.upper()
            if idx % 100 == 0:
                print(f"Processing {idx} out of {gene_info_table.shape[0]}")

        except Exception as e:
            missing_cnt += 1
            print(f"❌ request error for ：{ensembl_id}")
            print(e)
            continue

    expanded_gene_info_table.to_csv('data/expanded_gene_info_table.csv', header=True, index=False)
    print(f"success rate: {missing_cnt/total_cnt} missing: {missing_cnt} total: {total_cnt}")


map_gene_id()
