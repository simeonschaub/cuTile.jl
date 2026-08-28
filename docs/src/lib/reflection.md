# Code Inspection

```@meta
CurrentModule = cuTile
```

## Compiling a signature

```@docs
code_tiled
code_typed
code_ircode
code_structured
code_sass
code_ptx
```

## Jobs

```@docs
tile_job
```

## Intercepting a launch

```@docs
@device_code_tiled
@device_code_typed
@device_code_structured
```

cuTile compilations are GPUCompiler `CompilerJob`s, so CUDA's macros work on
`@cuda backend=cuTile` launches too: `CUDA.@device_code_typed`,
`CUDA.@device_code_ptx` (unstable, see [`code_ptx`](@ref)) and
`CUDA.@device_code_sass`, which intercepts module loads at the driver level
(via CUPTI) and shows the binary any backend actually loaded, with Julia
source locations.
