/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GLOW_EXPORTER_ONNXMODELWRITER_H
#define GLOW_EXPORTER_ONNXMODELWRITER_H

#include "glow/Exporter/CommonOperatorWriter.h"
#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "onnx/onnx_pb.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <list>
#include <string>

/// ONNX traits for protobuf types.
struct ONNX_TRAITS {
  using GraphProto = ONNX_NAMESPACE::GraphProto;
};

namespace glow {

/// Unique set of visited nodes.
using ReportedNodes = std::unordered_set<const Node *>;

/// Writes ONNX models.
class ONNXModelWriter : public CommonOperatorWriter<ONNX_TRAITS> {
  // Declare shorter aliases.
  using GraphType = typename ONNX_TRAITS::GraphProto;
  using NodeType = ONNX_NAMESPACE::NodeProto;
  using TensorType = ONNX_NAMESPACE::TensorProto;
  using AttrType = ONNX_NAMESPACE::AttributeProto;
  using ValueInfoType = ONNX_NAMESPACE::ValueInfoProto;

  // ModelProto that we are writing to.
  ONNX_NAMESPACE::ModelProto modelProto_;
  // GraphProto that we are writing to.
  ONNX_TRAITS::GraphProto *graphProto_;
  /// Current IR version of ONNX.
  const size_t irVersion_;
  /// Current version of ONNX standard.
  const size_t opsetVersion_;
  /// Keeps the track of already visited or processed nodes.
  ReportedNodes reportedNodes_;
  /// Whether we use zip mode or not
  const bool zipMode_;
  /// Whether we use text mode or not
  const bool textMode_;
  /// Whether to use custom ONNX ops.
  const bool useGlowCustomOps_;
  /// Whether we are writing a DAG.
  const bool dagMode_;
  /// A dedicated list of initializers in case the tensors get too big and don't
  /// fit into the model.
  std::list<TensorType> initializers_;
  /// Holds all Functions from a DAG that are being written when in dagMode_.
  llvm::SmallSet<Function *, 6> functionsFromDAG_;
  /// Writes tensor shape from placeholder \p PH into protpbuf \p valueProto.
  void tensorShapeFromPlaceholder(const Placeholder *PH,
                                  ValueInfoType *valueProto);
  /// Writes all inputs and outputs with operator name \p opName from give Node
  /// \p node into protobuf \p proto.
  static Error writeAllWithNode(const std::string &opName, const Node *node,
                                GraphType &graph, NodeType *proto);
  /// Writes all inputs and outputs with operator name \p opName from give Node
  /// \p node into created node protobuf using \p graph.
  static Error writeAll(const std::string &opName, const Node *node,
                        GraphType &graph);

  /// Add an initializer. Depending on \ref zipMode_, it will add directly to
  /// the \p graph or to a separate list.
  TensorType *addInitializer(GraphType &graph);

  /// Special case node writer for Glow convolutions with quantized inputs and
  /// outputs.
  Error writeTensorwiseQuantizedConvolution(const ConvolutionNode *node,
                                            GraphType &graph);

  /// Write \p node to \p graph using custom Glow Nodes, exported via
  /// auto-generated export logic in NodeGen.
  Error writeGlowCustomOperator(const Node *node, GraphType &graph);

  /// Setup a new proto \ref modelProto_ and \ref graphProto_.
  void setupNewProto();

  /// Write the current Function \ref F_ to \ref graphProto_. \returns if there
  /// was an issue during iteration or writing.
  Error writeFunction();

  /// Finalize the written function and write it out to \p filename. \returns if
  /// there is an error encountered.
  Error finalizeAndWriteProto(llvm::StringRef filename);

  /// Adds a metadata prop with \p key and \p val to \ref modelProto_.
  void addMetadataProp(const std::string &key, const std::string &val);

  /// Write out the Functions and metadata for all DAGNodes in \p postOrder
  /// given parent \p mod.
  Error writePartitionAndMetadataProps(
      Module &mod, llvm::ArrayRef<const runtime::DAGNode *> postOrder);

  /// \returns whether \p PH is an intermediate PH for the DAG being written
  /// (i.e. both input and an output for Functions in \ref functionsFromDAG_).
  bool isIntermediatePHForDAG(const Placeholder *PH);

public:
  /// Converts \p glowType to \p protoType.
  static typename TensorType::DataType convertType(const Type &glowType);
  /// Writes Glow tensor \p T to proto output \p out. Depending on
  /// \p useGlowCustomOps meta info will be annotated differently.
  static void writeTensor(const Tensor &T, TensorType *out,
                          bool useGlowCustomOps = false);

  /// Creates an ONNX model writer to serialize \p F graph into file
  /// \p modelFilename, writing \p irVersion and \p opsetVersion.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort. It also supports
  /// serialization with text format or binary format depending on \p textMode.
  /// If \p zipMode is true, it will save weights into individual TensorProto
  /// file along with the model file and package them into a zip file. If
  /// \p useGlowCustomOps then it will use auto-generated export logic via
  /// NodeGen to export all Glow Nodes as is via custom ops, instead of trying
  /// to abide by the official ONNX ops.
  ONNXModelWriter(const std::string &modelFilename, Function &F,
                  size_t irVersion, size_t opsetVersion,
                  Error *errPtr = nullptr, bool textMode = false,
                  bool zipMode = false, bool useGlowCustomOps = false);

  /// Creates an ONNX model writer to serialize \p dagList into file

  /// \p modelFilename, writing \p irVersion and \p opsetVersion. Each partition
  /// from \p dagList will be annotated with the name of the partition to the
  /// op. This exporter requires using \ref useGlowCustomOps_ and sets it true
  /// as such. If \p errPtr is not null then if an error occurs it will get
  /// assigned there otherwise if an error occurs it will abort. It also
  /// supports serialization with text format or binary format depending on
  /// \p textMode. If \p zipMode is true, it will save weights into individual
  /// TensorProto file along with the model file and package them into a zip
  /// file.
  ONNXModelWriter(const std::string &modelFilename, runtime::DAGListTy &dagList,
                  size_t irVersion, size_t opsetVersion,
                  Error *errPtr = nullptr, bool textMode = false,
                  bool zipMode = false);

private:
  /// \returns error for the unexpected node kind.
  static Error writeUnexpectedKind(const Node *node) {
    RETURN_ERR(strFormat("Glow can not export node %s, unsupported kind: %s.",
                         node->getName().str().c_str(), node->getKindName()));
  }

  /// Declares the overriden all pure virtual methods, declared in base class.
#define DEF_NODE(CLASS, NAME)                                                  \
  Error write##NAME(const CLASS *, GraphType &) override;
#include "glow/AutoGenNodes.def"
};

} // namespace glow

#endif // GLOW_EXPORTER_ONNXMODELWRITER_H
