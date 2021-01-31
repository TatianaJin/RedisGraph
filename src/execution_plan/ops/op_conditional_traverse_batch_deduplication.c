/*
* Copyright 2021 HDL
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "op_conditional_traverse_batch_deduplication.h"

#include <stdio.h>

#include "../../query_ctx.h"
#include "../../util/qsort.h"
#include "../../util/simple_timer.h"
#include "shared/print_functions.h"

#define ISLT(a, b) \
  (ENTITY_GET_ID(Record_GetNode(*a, op->srcNodeIdx)) < ENTITY_GET_ID(Record_GetNode(*b, op->srcNodeIdx)))

inline bool Traverse_SetEdgeReuse(OpCondTraverseBD *op) {
  EdgeTraverseCtx *edge_ctx = op->edge_ctx;
  // Return false if all edges have been consumed.
  if (op->edge_size == 0) return false;

  ASSERT(op->edge_size > 0);
  // Pop an edge and add it to the Record.
  Edge e = edge_ctx->edges[--op->edge_size];
  Record_AddEdge(op->r, edge_ctx->edgeRecIdx, e);
  return true;
}

/* Forward declarations. */
static OpResult CondTraverseInitBD(OpBase *opBase);
static Record CondTraverseConsumeBD(OpBase *opBase);
static Record CondTraverseConsumeBDProfile(OpBase *opBase);
static OpResult CondTraverseResetBD(OpBase *opBase);
static OpBase *CondTraverseCloneBD(const ExecutionPlan *plan, const OpBase *opBase);
static void CondTraverseFreeBD(OpBase *opBase);

static int CondTraverseToStringBD(const OpBase *ctx, char *buf, uint buf_len) {
  return TraversalToString(ctx, buf, buf_len, ((const OpCondTraverseBD *)ctx)->ae);
}

static void _FirstTraversalInit(OpCondTraverseBD *op) {  // copied from cond traverse
  // Create both filter and result matrices.
  size_t required_dim = Graph_RequiredMatrixDim(op->graph);
  GrB_Matrix_new(&op->M, GrB_BOOL, op->record_cap, required_dim);
  GrB_Matrix_new(&op->F, GrB_BOOL, op->record_cap, required_dim);

  // Prepend the filter matrix to algebraic expression as the leftmost operand.
  AlgebraicExpression_MultiplyToTheLeft(&op->ae, op->F);

  // Optimize the expression tree.
  AlgebraicExpression_Optimize(&op->ae);
}

static void _PopulateFilterMatrixProfiled(OpCondTraverseBD *op) {
  // sort the record array
  double tic[2];
  simple_tic(tic);
  QSORT(Record, op->records, op->record_count, ISLT);

  // find distinct src nodes for traversal
  uint64_t row_idx = 0;
  NodeID current_src_id = INVALID_ENTITY_ID;
  for (uint i = 0; i < op->record_count; i++) {
    Record r = op->records[i];
    /* Update filter matrix F, set row row_idx at position srcId
     * F[row_idx, srcId] = true. */
    NodeID srcId = ENTITY_GET_ID(Record_GetNode(r, op->srcNodeIdx));
    if (srcId != current_src_id) {
      ASSERT(srcId > current_src_id);
      op->record_offset[row_idx] = i;
      GrB_Matrix_setElement_BOOL(op->F, true, row_idx, srcId);
      ++row_idx;
      current_src_id = srcId;
    }
  }
  op->record_offset[row_idx] = op->record_count;
  op->op.stats->sortTime += simple_toc(tic);

  /* Verification experiments: count redundancy in source nodes
   * output format: (in binary, uint64_t)
   * <num_records><batch_i_record_1_source_id><batch_i_record_2_source_id>...
   * <num_records><batch_i+1_record_1_source_id><batch_i+1_record_2_source_id>...  */
  simple_tic(tic);
  fwrite(&row_idx, sizeof(uint64_t), 1, op->stats_file);
  for (uint i = 0; i < row_idx; ++i) {
    Record r = op->records[op->record_offset[i]];
    NodeID srcId = ENTITY_GET_ID(Record_GetNode(r, op->srcNodeIdx));
    fwrite(&srcId, sizeof(NodeID), 1, op->stats_file);
  }
  op->op.stats->statsTime += simple_toc(tic);
  /* redisgraphG */
}
static void _PopulateFilterMatrix(OpCondTraverseBD *op) {
  // sort the record array
  QSORT(Record, op->records, op->record_count, ISLT);

  // find distinct src nodes for traversal
  uint64_t row_idx = 0;
  NodeID current_src_id = INVALID_ENTITY_ID;
  for (uint i = 0; i < op->record_count; i++) {
    Record r = op->records[i];
    /* Update filter matrix F, set row row_idx at position srcId
     * F[row_idx, srcId] = true. */
    NodeID srcId = ENTITY_GET_ID(Record_GetNode(r, op->srcNodeIdx));
    if (srcId != current_src_id) {
      ASSERT(srcId > current_src_id);
      op->record_offset[row_idx] = i;
      GrB_Matrix_setElement_BOOL(op->F, true, row_idx, srcId);
      ++row_idx;
      current_src_id = srcId;
    }
  }
  op->record_offset[row_idx] = op->record_count;

  /* Verification experiments: count redundancy in source nodes
   * output format: (in binary, uint64_t)
   * <num_records><batch_i_record_1_source_id><batch_i_record_2_source_id>...
   * <num_records><batch_i+1_record_1_source_id><batch_i+1_record_2_source_id>...  */
  fwrite(&row_idx, sizeof(uint64_t), 1, op->stats_file);
  for (uint i = 0; i < row_idx; ++i) {
    Record r = op->records[op->record_offset[i]];
    NodeID srcId = ENTITY_GET_ID(Record_GetNode(r, op->srcNodeIdx));
    fwrite(&srcId, sizeof(NodeID), 1, op->stats_file);
  }
  /* redisgraphG */
}

static void _TraverseProfiled(OpCondTraverseBD *op) {
  // If op->F is null, this is the first time we are traversing.
  // Create both filter and result matrices and prepend filter matrix as the left most operand.
  if (op->F == GrB_NULL) {
    _FirstTraversalInit(op);
  }

  // In batch deduplication mode, we populate the filter matrix for OpGCondTraverse in a different way from that for
  // OpCondTraverse. We populate the filter for each distinct source node (i.e. each record group) instead of for each
  // record.
  _PopulateFilterMatrixProfiled(op);

  // Evaluate expression.
  double tic[2];
  simple_tic(tic);

  AlgebraicExpression_Eval(op->ae, op->M);
  if (op->iter == NULL) {
    GxB_MatrixTupleIter_new(&op->iter, op->M);
  } else {
    GxB_MatrixTupleIter_reuse(op->iter, op->M);
  }
  // force no pending
  GrB_Index nvals;
  GrB_Info res = GrB_Matrix_nvals(&nvals, op->M);

  op->op.stats->aeEvalTime += simple_toc(tic);

  // Clear filter matrix.
  GrB_Matrix_clear(op->F);
}

static void _Traverse(OpCondTraverseBD *op) {
  // If op->F is null, this is the first time we are traversing.
  // Create both filter and result matrices and prepend filter matrix as the left most operand.
  if (op->F == GrB_NULL) {
    _FirstTraversalInit(op);
  }

  // In batch deduplication mode, we populate the filter matrix for OpGCondTraverse in a different way from that for
  // OpCondTraverse. We populate the filter for each distinct source node (i.e. each record group) instead of for each
  // record.
  _PopulateFilterMatrix(op);

  // Evaluate expression.
  AlgebraicExpression_Eval(op->ae, op->M);
  if (op->iter == NULL) {
    GxB_MatrixTupleIter_new(&op->iter, op->M);
  } else {
    GxB_MatrixTupleIter_reuse(op->iter, op->M);
  }

  // Clear filter matrix.
  GrB_Matrix_clear(op->F);
}

OpBase *NewCondTraverseOpBD(const ExecutionPlan *plan, Graph *g, AlgebraicExpression *ae) {
  OpCondTraverseBD *op = rm_malloc(sizeof(OpCondTraverseBD));
  op->graph = g;
  op->ae = ae;
  op->r = NULL;
  op->iter = NULL;
  op->F = GrB_NULL;
  op->M = GrB_NULL;
  op->records = NULL;
  op->record_count = 0;
  op->edge_ctx = NULL;
  op->dest_label = NULL;
  op->record_cap = QueryCtx_GetBatchSize();
  op->dest_label_id = GRAPH_NO_LABEL;

  // Set our Op operations
  OpBase_Init((OpBase *)op, OPType_CONDITIONAL_TRAVERSE_BD, "Conditional Traverse BD", CondTraverseInitBD,
              CondTraverseConsumeBD, CondTraverseResetBD, CondTraverseToStringBD, CondTraverseCloneBD,
              CondTraverseFreeBD, false, plan);
  ((OpBase *)op)->profile = CondTraverseConsumeBDProfile;

  bool aware = OpBase_Aware((OpBase *)op, AlgebraicExpression_Source(ae), &op->srcNodeIdx);
  UNUSED(aware);
  ASSERT(aware == true);

  const char *dest = AlgebraicExpression_Destination(ae);
  op->destNodeIdx = OpBase_Modifies((OpBase *)op, dest);
  // Check the QueryGraph node and retrieve label data if possible.
  QGNode *dest_node = QueryGraph_GetNodeByAlias(plan->query_graph, dest);
  op->dest_label = dest_node->label;
  op->dest_label_id = dest_node->labelID;

  const char *edge = AlgebraicExpression_Edge(ae);
  if (edge) {
    /* This operation will populate an edge in the Record.
     * Prepare all necessary information for collecting matching edges. */
    uint edge_idx = OpBase_Modifies((OpBase *)op, edge);
    QGEdge *e = QueryGraph_GetEdgeByAlias(plan->query_graph, edge);
    op->edge_ctx = Traverse_NewEdgeCtx(ae, e, edge_idx);
  }

  return (OpBase *)op;
}

static OpResult CondTraverseInitBD(OpBase *opBase) {
  OpCondTraverseBD *op = (OpCondTraverseBD *)opBase;
  // Create 'records' with this Init function as 'record_cap'
  // might be set during optimization time (applyLimit)
  // If cap greater than BATCH_SIZE is specified,
  // use BATCH_SIZE as the value.
  if (op->record_cap > QueryCtx_GetBatchSize()) op->record_cap = QueryCtx_GetBatchSize();
  op->records = rm_calloc(op->record_cap, sizeof(Record));
  op->record_offset = rm_calloc(op->record_cap + 1, sizeof(uint));

  if (op->op.parent->type == OPType_CONDITIONAL_TRAVERSE_BD) {
    // TODO(tatiana): consider outputing records ordered by parent->srcNodeIdx for deduplication in parent op
    // now outputing records following the result order
  }

  if (op->stats_file == NULL) {
    char file_name[40];
    snprintf(file_name, 40, "/home/tati/g_cond_traverse_%d", op->destNodeIdx);
    op->stats_file = fopen(file_name, "wb");
    uint64_t size = 0;  // write 0 to reset
    fwrite(&size, sizeof(uint64_t), 1, op->stats_file);
  }

  return OP_OK;
}

static Record CondTraverseConsumeBDProfile(OpBase *opBase) {
  OpCondTraverseBD *op = (OpCondTraverseBD *)opBase;
  OpBase *child = op->op.children[0];

  /* If we're required to update an edge and have one queued, we can return early.
   * If the current record group is not yet depleted, get the next record and reuse collected edges as needed.
   * Otherwise, try to get a new pair of source and destination nodes. */
  if (op->edge_ctx && Traverse_SetEdgeReuse(op)) return OpBase_CloneRecordTimed(op->r, op->op.stats);
  if (++op->group_idx < op->group_end) {
    op->r = op->records[op->group_idx];
    Record_AddNode(op->r, op->destNodeIdx, op->dest_node);

    if (op->edge_ctx) {
      op->edge_size = array_len(op->edge_ctx->edges);
      Traverse_SetEdgeReuse(op);
    }
    return OpBase_CloneRecordTimed(op->r, op->op.stats);
  }

  // FIXME(tatiana): enumerate from group

  bool depleted = true;
  NodeID src_id = INVALID_ENTITY_ID;
  NodeID dest_id = INVALID_ENTITY_ID;

  while (true) {
    if (op->iter) GxB_MatrixTupleIter_next(op->iter, &src_id, &dest_id, &depleted);

    // Manage to get a result to return
    if (!depleted) break;

    /* Run out of tuples, try to get new data.
     * Free old records. */
    op->r = NULL;
    for (uint i = 0; i < op->record_count; i++) OpBase_DeleteRecord(op->records[i]);

    // Ask child operations for data.
    for (op->record_count = 0; op->record_count < op->record_cap; op->record_count++) {
      Record childRecord = OpBase_Consume(child);
      // If the Record is NULL, the child has been depleted.
      if (!childRecord) break;
      if (!Record_GetNode(childRecord, op->srcNodeIdx)) {
        /* The child Record may not contain the source node in scenarios like
         * a failed OPTIONAL MATCH. In this case, delete the Record and try again. */
        OpBase_DeleteRecord(childRecord);
        op->record_count--;
        continue;
      }

      // Store received record.
      Record_PersistScalars(childRecord);
      op->records[op->record_count] = childRecord;
    }

    // No data.
    if (op->record_count == 0) {
      printf("no data\n");
      return NULL;
    }

    _TraverseProfiled(op);
  }

  // Get the group of records with the same source node
  op->group_idx = op->record_offset[src_id];
  op->group_end = op->record_offset[src_id + 1];
  op->r = op->records[op->group_idx];

  op->dest_node = GE_NEW_LABELED_NODE(op->dest_label, op->dest_label_id);
  Graph_GetNode(op->graph, dest_id, &op->dest_node);
  Record_AddNode(op->r, op->destNodeIdx, op->dest_node);

  if (op->edge_ctx) {
    Node *srcNode = Record_GetNode(op->r, op->srcNodeIdx);
    // Collect all appropriate edges connecting the current pair of endpoints.
    Traverse_ResetEdgeCtx(op->edge_ctx);  // clear edges for previous src-dest pair
    Traverse_CollectEdges(op->edge_ctx, ENTITY_GET_ID(srcNode), ENTITY_GET_ID(&op->dest_node));
    op->edge_size = array_len(op->edge_ctx->edges);
    // We're guaranteed to have at least one edge.
    Traverse_SetEdgeReuse(op);
  }

  return OpBase_CloneRecordTimed(op->r, op->op.stats);
}

static Record CondTraverseConsumeBD(OpBase *opBase) {
  OpCondTraverseBD *op = (OpCondTraverseBD *)opBase;
  OpBase *child = op->op.children[0];

  /* If we're required to update an edge and have one queued, we can return early.
   * If the current record group is not yet depleted, get the next record and reuse collected edges as needed.
   * Otherwise, try to get a new pair of source and destination nodes. */
  if (op->edge_ctx && Traverse_SetEdgeReuse(op)) return OpBase_CloneRecord(op->r);

  // enumerate from record group by source node
  if (++op->group_idx < op->group_end) {
    op->r = op->records[op->group_idx];
    Record_AddNode(op->r, op->destNodeIdx, op->dest_node);

    if (op->edge_ctx) {
      op->edge_size = array_len(op->edge_ctx->edges);
      Traverse_SetEdgeReuse(op);
    }
    return OpBase_CloneRecord(op->r);
  }

  bool depleted = true;
  NodeID src_id = INVALID_ENTITY_ID;
  NodeID dest_id = INVALID_ENTITY_ID;

  while (true) {
    if (op->iter) GxB_MatrixTupleIter_next(op->iter, &src_id, &dest_id, &depleted);

    // Manage to get a result to return
    if (!depleted) break;

    /* Run out of tuples, try to get new data.
     * Free old records. */
    op->r = NULL;
    for (uint i = 0; i < op->record_count; i++) OpBase_DeleteRecord(op->records[i]);

    // Ask child operations for data.
    for (op->record_count = 0; op->record_count < op->record_cap; op->record_count++) {
      Record childRecord = OpBase_Consume(child);
      // If the Record is NULL, the child has been depleted.
      if (!childRecord) break;
      if (!Record_GetNode(childRecord, op->srcNodeIdx)) {
        /* The child Record may not contain the source node in scenarios like
         * a failed OPTIONAL MATCH. In this case, delete the Record and try again. */
        OpBase_DeleteRecord(childRecord);
        op->record_count--;
        continue;
      }

      // Store received record.
      Record_PersistScalars(childRecord);
      op->records[op->record_count] = childRecord;
    }

    // No data.
    if (op->record_count == 0) {
      printf("no data\n");
      return NULL;
    }

    _Traverse(op);
  }

  // Get the group of records with the same source node
  op->group_idx = op->record_offset[src_id];
  op->group_end = op->record_offset[src_id + 1];
  op->r = op->records[op->group_idx];

  op->dest_node = GE_NEW_LABELED_NODE(op->dest_label, op->dest_label_id);
  Graph_GetNode(op->graph, dest_id, &op->dest_node);
  Record_AddNode(op->r, op->destNodeIdx, op->dest_node);

  if (op->edge_ctx) {
    Node *srcNode = Record_GetNode(op->r, op->srcNodeIdx);
    // Collect all appropriate edges connecting the current pair of endpoints.
    Traverse_ResetEdgeCtx(op->edge_ctx);  // clear edges for previous src-dest pair
    Traverse_CollectEdges(op->edge_ctx, ENTITY_GET_ID(srcNode), ENTITY_GET_ID(&op->dest_node));
    op->edge_size = array_len(op->edge_ctx->edges);
    // We're guaranteed to have at least one edge.
    Traverse_SetEdgeReuse(op);
  }

  return OpBase_CloneRecord(op->r);
}

static OpResult CondTraverseResetBD(OpBase *ctx) {
  OpCondTraverseBD *op = (OpCondTraverseBD *)ctx;

  // Do not explicitly free op->r, as the same pointer is also held
  // in the op->records array and as such will be freed there.
  op->r = NULL;
  for (uint i = 0; i < op->record_count; i++) OpBase_DeleteRecord(op->records[i]);
  op->record_count = 0;

  if (op->edge_ctx) Traverse_ResetEdgeCtx(op->edge_ctx);

  if (op->iter) {
    GxB_MatrixTupleIter_free(op->iter);
    op->iter = NULL;
  }
  if (op->F != GrB_NULL) GrB_Matrix_clear(op->F);

  // reset stats collection
  uint64_t size = 0;  // write 0 to reset
  fwrite(&size, sizeof(uint64_t), 1, op->stats_file);

  return OP_OK;
}

static inline OpBase *CondTraverseCloneBD(const ExecutionPlan *plan, const OpBase *opBase) {
  ASSERT(opBase->type == OPType_CONDITIONAL_TRAVERSE_BD);
  OpCondTraverseBD *op = (OpCondTraverseBD *)opBase;
  return NewCondTraverseOpBD(plan, QueryCtx_GetGraph(), AlgebraicExpression_Clone(op->ae));
}

static void CondTraverseFreeBD(OpBase *ctx) {
  OpCondTraverseBD *op = (OpCondTraverseBD *)ctx;

  if (op->iter) {
    GxB_MatrixTupleIter_free(op->iter);
    op->iter = NULL;
  }

  if (op->F != GrB_NULL) {
    GrB_Matrix_free(&op->F);
    op->F = GrB_NULL;
  }

  if (op->M != GrB_NULL) {
    GrB_Matrix_free(&op->M);
    op->M = GrB_NULL;
  }

  if (op->ae) {
    AlgebraicExpression_Free(op->ae);
    op->ae = NULL;
  }

  if (op->edge_ctx) {
    Traverse_FreeEdgeCtx(op->edge_ctx);
    op->edge_ctx = NULL;
  }

  if (op->records) {
    for (uint i = 0; i < op->record_count; i++) OpBase_DeleteRecord(op->records[i]);
    rm_free(op->records);
    op->records = NULL;
  }

  if (op->record_offset) {
    rm_free(op->record_offset);
    op->record_offset = NULL;
  }

  if (op->stats_file) {
    fclose(op->stats_file);
  }
}
