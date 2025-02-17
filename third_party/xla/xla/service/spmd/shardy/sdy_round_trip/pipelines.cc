/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"

#include <cassert>

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/round_trip_common/pipeline_passes.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_ops.h"
#include "xla/service/spmd/shardy/sdy_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/sdy_round_trip/import_shardings.h"

namespace xla {
namespace sdy {

using ::mlir::PassPipelineRegistration;

void addSdyRoundTripExportPipeline(mlir::OpPassManager& pm) {
  // NOTE: we don't do any exporting for ManualComputationOp, since during
  // SDY round-trip we expect the same pattern of custom calls to continue to
  // exist. We save `sdy.sharding`s on those custom calls during
  // `createSdyRoundTripExportShardingsPass` and make use of
  // `createSdyRoundTripImportShardingsPass` to import them.
  pm.addPass(createSdyRoundTripExportOpsPass());
  pm.addPass(createSdyRoundTripExportShardingsPass());
}

void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm) {
  addCommonPreImportPasses(pm);
  pm.addPass(createSdyRoundTripImportShardingsPass());
  addCommonPostImportPasses(pm);
}

void registerSdyRoundTripExportPipeline() {
  PassPipelineRegistration<> exportPipeline(
      "xla-sdy-round-trip-export-pipeline",
      "Run passes to export the SDY (Shardy) dialect into an MHLO module, "
      "but with the SDY ops/attrs saved for roundtripping.",
      addSdyRoundTripExportPipeline);
}

void registerSdyRoundTripImportPipeline() {
  PassPipelineRegistration<> importPipeline(
      "xla-sdy-round-trip-import-pipeline",
      "Run passes to import an mhlo module into the SDY (Shardy) dialect.",
      addSdyRoundTripImportPipeline);
}

}  // namespace sdy
}  // namespace xla
