"""
Compute the metrics for the retrieval task
"""
class Metrics:
    def __init__(self):
        pass

    def computeMRR(self, ranks_data, rank_column_name):
        """
        ranks_data = all the issues for the repository
        """
        unique_queries=ranks_data['q_id'].unique() ## get unique queries

        rr_total = 0
        mrr = 0

        for query in unique_queries:
            query_data = ranks_data.query("q_id == @query")
            ## sort by assending based on rank column
            query_data = query_data.sort_values(by=rank_column_name, ascending=True)
            ## filter the c_is_gt=1 and get the rank of it

            rr = 0
            ## check if there are at least one relevant doc retrieved
            if query_data.query("c_is_gt == 1").shape[0] > 0:
                first_relevant_issue = query_data.query("c_is_gt == 1")[rank_column_name].min()
            
                ## SABD - testing (Consider only the olderest corpus_issue)
                ## first_relevant_issue = query_data.query("c_is_gt == 1").sort_values("c_id").head(1)['rank'].min()

                ## compute the reciprocal of the first relevant issue
                rr = 1/first_relevant_issue
            rr_total += rr

        mrr = rr_total/unique_queries.shape[0]

        return mrr
    
    # def computeMAP(self, ranks_data):
    #     """
    #     ranks_data = all the issues for the repository
    #     """
    #     unique_queries=ranks_data['q_id'].unique()
    #     ap_total = 0 ## total of per query average precisions
    #     map = 0

    #     for query in unique_queries:
    #         query_data = ranks_data.query("q_id == @query")
    #         query_data = query_data.sort_values(by='rank', ascending=True)
    #         relevant_issues = query_data.query("c_is_gt == 1")

    #         ## SABD - testing (Consider only the olderest corpus_issue)
    #         # relevant_issues = query_data.query("c_is_gt == 1").sort_values("c_id").head(1)


    #         relevant_issues_count = relevant_issues.shape[0]

    #         if relevant_issues_count > 0:
    #             per_query_precision = 0
    #             ranks = relevant_issues['rank'].values
    #             for i in range(relevant_issues_count):
    #                 per_query_precision += (i+1)/ranks[i]
    #             avg_precision_query = per_query_precision/relevant_issues_count
    #             ap_total += avg_precision_query

    #     map = ap_total/unique_queries.shape[0]

    #     return map

    def computeMAP(self, ranks_data, rank_column_name):
        """
        Compute Mean Average Precision (MAP) over all queries.

        Args:
            ranks_data (pd.DataFrame): DataFrame containing at least:
                - 'q_id': query ID
                - 'rank': rank assigned to each retrieved item (lower is better)
                - 'c_is_gt': 1 if item is relevant, 0 otherwise

        Returns:
            float: MAP score
        """
        unique_queries = ranks_data['q_id'].unique()
        ap_total = 0  # total of average precision over all queries

        for query in unique_queries:
            query_data = ranks_data[ranks_data['q_id'] == query]
            query_data = query_data.sort_values(by=rank_column_name, ascending=True)
            relevant_mask = query_data['c_is_gt'] == 1

            if relevant_mask.sum() == 0:
                continue  # skip queries with no relevant items

            precisions = []
            retrieved_relevant = 0

            for i, (_, row) in enumerate(query_data.iterrows(), start=1):  # i is position in ranking
                if row['c_is_gt'] == 1:
                    retrieved_relevant += 1
                    precisions.append(retrieved_relevant / i)

            avg_precision = sum(precisions) / relevant_mask.sum()
            ap_total += avg_precision

        map_score = ap_total / len(unique_queries)
        return map_score

    
    def computeRecall_K(self, ranks_data, k, rank_column_name):
        """
        ranks_data = all the issues for the repository
        """
        unique_queries=ranks_data['q_id'].unique()
        recall_k_total = 0
        avg_recall_k = 0

        for query in unique_queries:
            recall_k = 0
            query_data = ranks_data.query("q_id == @query")
            query_data = query_data.sort_values(by=rank_column_name, ascending=True)
            relevant_issues = query_data.query("c_is_gt == 1")
            
            ## filter for the top k issues
            top_k_retreval_data = query_data.head(k)
            
            total_relevant_issues_count = relevant_issues.shape[0]
            relevant_issues_in_top_k = top_k_retreval_data.query("c_is_gt == 1").shape[0]

            if relevant_issues_in_top_k > 0:
                recall_k = relevant_issues_in_top_k / total_relevant_issues_count
                recall_k_total += recall_k
            
        avg_recall_k = recall_k_total/unique_queries.shape[0]

        return avg_recall_k