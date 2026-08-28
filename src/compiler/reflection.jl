#=============================================================================
 Reflection utilities
=============================================================================#

export code_tiled
public code_typed, code_ircode, code_structured
public code_ptx, code_sass

function disassemble_tileir(bytecode::Vector{UInt8}, version::VersionNumber;
                            debuginfo::Bool=false)::String
    validate_bytecode_version(version)
    disassembler_version = tileir_disassembler_version()
    version <= disassembler_version || throw(ArgumentError(
        "Tile IR bytecode v$version cannot be decoded by the selected v$disassembler_version " *
        "disassembler"))
    disasm = tileir_disassembler(; debuginfo)
    mktempdir() do dir
        input_path = joinpath(dir, "kernel.tile")
        write(input_path, bytecode)
        read(`$disasm $input_path`, String)
    end
end

"""
    code_ircode(mi::MethodInstance; world, always_inline=true) -> (IRCode, rettype)

Get optimized IRCode for a MethodInstance using cuTile's overlay method table.
If always_inline=true (default), forces all functions to be inlined.
"""
function code_ircode(mi::MethodInstance; world::UInt=Base.get_world_counter(),
                     always_inline::Bool=true)
    cache = CacheView{CuTileResults}(:cuTile, world)
    interp = cuTileInterpreter(cache; always_inline)
    result = CC.typeinf_ircode(interp, mi, nothing)

    if result === nothing
        throw(ErrorException("Type inference failed for $mi"))
    end

    ir, rettype = result
    return ir, rettype
end

"""
    process_const_argtypes(f, argtypes) -> (stripped, const_argtypes)

Split `Constant{T,V}` types from argtypes for method lookup, and build a
`const_argtypes` vector with `CC.Const(V)` entries for const-seeded inference.

Returns `(stripped, nothing)` when no Constant types are present.
"""
function process_const_argtypes(@nospecialize(f), @nospecialize(argtypes))
    params = argtypes isa DataType ? argtypes.parameters :
             argtypes isa Tuple ? argtypes : fieldtypes(argtypes)
    has_consts = any(T -> T <: Constant || CC.isconstType(T), params)
    stripped_params = map(params) do T
        T <: Constant ? constant_eltype(T) : T
    end
    stripped = Tuple{stripped_params...}
    const_argtypes = if has_consts
        cats = Any[CC.Const(f)]
        for T in params
            if T <: Constant
                push!(cats, CC.Const(constant_value(T)))
            elseif CC.isconstType(T)
                push!(cats, CC.Const(T.parameters[1]))
            else
                push!(cats, T)
            end
        end
        cats
    else
        nothing
    end
    return stripped, const_argtypes
end

constant_eltype(::Type{Constant{T,V}}) where {T,V} = T
constant_value(::Type{Constant{T,V}}) where {T,V} = V


#=============================================================================
 Jobs
=============================================================================#

public tile_job

"""
    tile_job(f, argtypes; sm_arch, opt_level, num_ctas, occupancy, num_worker_warps,
             bytecode_version, world) -> CompilerJob

The `CompilerJob` for compiling `f` on `argtypes` (which may contain
`Constant{T,V}` types) with the given compilation options. `sm_arch` may be
`nothing` for stages that do not run `tileiras`.
"""
function tile_job(@nospecialize(f), @nospecialize(argtypes);
                  world::UInt=Base.get_world_counter(), kwargs...)
    stripped, const_argtypes = process_const_argtypes(f, argtypes)
    mi = lookup_method_instance(f, stripped; world)
    tile_job(mi, world; const_argtypes, kwargs...)
end

# `(f, tt)` with `Constant` argument types restored, for printing headers.
function job_signature(job::TileJob)
    mi = job.source
    ftype = mi.specTypes.parameters[1]
    f = isdefined(ftype, :instance) ? ftype.instance : ftype
    arg_types = collect(Any, mi.specTypes.parameters[2:end])
    cats = job.config.params.const_argtypes
    if cats !== nothing
        # cats is [Const(f), arg2, ...]; arg_types omits f.
        for i in eachindex(arg_types)
            cats[i+1] isa CC.Const && (arg_types[i] = typeof(Constant(cats[i+1].val)))
        end
    end
    return f, Tuple{arg_types...}
end

reflection_cache(job::TileJob) = CacheView{CuTileResults}(:cuTile, job.world)

# The architecture tileiras will target: the job's, or the active device's.
function resolve_target(job::TileJob, purpose::String)
    sm_arch = job.config.target.sm_arch
    sm_arch === nothing || return sm_arch
    try
        default_sm_arch()
    catch
        throw(ArgumentError("sm_arch must be specified when $purpose without a CUDA device"))
    end
end

with_target(job::TileJob, sm_arch::VersionNumber) =
    CompilerJob(job; config=CompilerConfig(job.config;
        target=TileCompilerTarget(sm_arch, job.config.target.bytecode_version)))

# tileiras's -O level: the job's, else the kernel's hint, else 3.
resolve_opt_level(job::TileJob, kernel_meta, target) =
    something(resolve_hint(job.config.params.opt_level, kernel_meta, :opt_level, target), 3)

GPUCompiler.get_interpreter(job::TileJob) =
    cuTileInterpreter(reflection_cache(job); always_inline=job.config.always_inline)


#=============================================================================
 Stages
=============================================================================#

"""
    code_typed(f, argtypes; world, kwargs...) -> Vector{Any}

Return typed code for a cuTile function. Analogous to `Base.code_typed`.
"""
function code_typed(job::TileJob)
    ir, rettype = emit_julia(reflection_cache(job), job.source;
                             const_argtypes=job_const_argtypes(job))
    [ir => rettype]
end
code_typed(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_typed(tile_job(f, argtypes; kwargs...))
GPUCompiler.code_typed(job::TileJob; kwargs...) = code_typed(job)

"""
    code_structured(f, argtypes; kwargs...) -> Vector{Pair{StructuredIRCode, DataType}}

Return the structured IR for a cuTile function.
"""
function code_structured(job::TileJob; optimize::Bool=true)
    ir, rettype = emit_julia(reflection_cache(job), job.source;
                             const_argtypes=job_const_argtypes(job))
    sci, rettype, _ = emit_structured(ir, rettype)
    if optimize
        sci = copy(sci)
        run_passes!(sci)
    end
    [sci => rettype]
end
code_structured(@nospecialize(f), @nospecialize(argtypes); optimize::Bool=true, kwargs...) =
    code_structured(tile_job(f, argtypes; kwargs...); optimize)

# Uncached front half shared by the bytecode-and-beyond stages.
function emit_bytecode(job::TileJob)
    cache = reflection_cache(job)
    const_argtypes = job_const_argtypes(job)
    ir, rettype = emit_julia(cache, job.source; const_argtypes)
    sci, rettype, kernel_meta = emit_structured(ir, rettype)
    bytecode = emit_tile(sci, rettype, kernel_meta;
                         name=job.config.name, opts=job_opts(job), cache, const_argtypes)
    return (; bytecode, kernel_meta)
end

"""
    code_tiled([io::IO], f, argtypes; sm_arch, opt_level, num_ctas, occupancy,
               num_worker_warps, remarks=false)

Print the CUDA Tile IR for a Julia function as a textual MLIR representation.
Analogous to `code_llvm`.

Set `remarks=true` to also run `tileiras` and print its optimization remarks.
This requires `tileiras` 13.4 or newer. When no GPU is available, pass `sm_arch`
explicitly.
"""
function code_tiled(io::IO, job::TileJob; debuginfo::Bool=false, remarks::Bool=false)
    (; bytecode, kernel_meta) = emit_bytecode(job)
    bytecode_version = job.config.target.bytecode_version
    print(io, disassemble_tileir(bytecode, bytecode_version; debuginfo))
    if remarks
        tileiras_version() >= v"13.4" || throw(ArgumentError(
            "tileiras optimization remarks require tileiras 13.4 or newer"))
        validate_tileiras_target(bytecode_version)
        target = resolve_target(job, "requesting remarks")
        _, text = run_tileiras(bytecode, target, resolve_opt_level(job, kernel_meta, target);
                               remarks=true)
        if !isempty(text)
            println(io)
            println(io, "// tileiras optimization remarks")
            for line in eachline(IOBuffer(text); keep=true)
                print(io, "// ", line)
            end
        end
    end
end
code_tiled(io::IO, @nospecialize(f), @nospecialize(argtypes);
           debuginfo::Bool=false, remarks::Bool=false, kwargs...) =
    code_tiled(io, tile_job(f, argtypes; kwargs...); debuginfo, remarks)
code_tiled(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_tiled(stdout, f, argtypes; kwargs...)

# GPUCompiler's stage API for a Tile job: `:obj` is the CUBIN, `:asm` the PTX
# it embeds. Uncached, like the other reflection entry points.
function GPUCompiler.compile(target::Symbol, job::TileJob)
    target === :llvm &&
        error("cuTile kernels have no LLVM stage; use `code_tiled` for the Tile IR")
    target in (:asm, :obj) || error("Unknown compilation output $target")
    job = with_target(job, resolve_target(job, "compiling"))
    validate_tileiras_target(job.config.target.bytecode_version)
    (; bytecode, kernel_meta) = emit_bytecode(job)
    sm_arch = job.config.target.sm_arch
    cubin, _ = run_tileiras(bytecode, sm_arch, resolve_opt_level(job, kernel_meta, sm_arch))
    meta = (; entry=job.config.name)
    return target === :obj ? (cubin, meta) : (extract_ptx(cubin), meta)
end

"""
    code_ptx([io::IO], f, argtypes; sm_arch, opt_level, num_ctas, occupancy,
             num_worker_warps)

Print the PTX that `tileiras` generates for a Julia function. This shows the
thread-level SIMT program the tile-level kernel is lowered to, with every
compiler decision (thread mapping, CTA size, pipelining, synchronization)
already made. When no GPU is available, pass `sm_arch` explicitly.

!!! warning "Unstable"
    The PTX is recorded by `tileiras` in an undocumented CUBIN section and may
    go away.
"""
code_ptx(io::IO, job::TileJob) = print(io, first(GPUCompiler.compile(:asm, job)))
code_ptx(io::IO, @nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_ptx(io, tile_job(f, argtypes; kwargs...))
code_ptx(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_ptx(stdout, f, argtypes; kwargs...)

# PTX is what `code_native` means for a Tile job (`CUDA.@device_code_ptx`).
GPUCompiler.code_native(io::IO, job::TileJob; kwargs...) = code_ptx(io, job)
GPUCompiler.code_llvm(io::IO, job::TileJob; kwargs...) =
    error("cuTile kernels have no LLVM stage; use `code_tiled` for the Tile IR")

"""
    code_sass([io::IO], f, argtypes; sm_arch, opt_level, num_ctas, occupancy,
              num_worker_warps)

Print the SASS machine code that a Julia function compiles to, by assembling
the Tile IR with `tileiras` and disassembling the resulting CUBIN with
`nvdisasm`. When no GPU is available, pass `sm_arch` explicitly. For the
binary a launch actually loaded, use `CUDA.@device_code_sass`.
"""
code_sass(io::IO, job::TileJob) = print(io, disassemble_cubin(first(GPUCompiler.compile(:obj, job))))
code_sass(io::IO, @nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_sass(io, tile_job(f, argtypes; kwargs...))
code_sass(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_sass(stdout, f, argtypes; kwargs...)


#=============================================================================
 Device code reflection macros
=============================================================================#

export @device_code_tiled
public @device_code_typed, @device_code_structured

function tile_hook(inner)
    function (job; io::IO=stdout, kwargs...)
        # The hook sees every back-end's compilations; only Tile jobs are ours.
        job isa TileJob || return
        f, tt = job_signature(job)
        println(io, "// $f($(join(tt.parameters, ", ")))")
        println(io)
        inner(io, job; kwargs...)
        println(io)
    end
end

"""
    @device_code_tiled [io=stdout] [remarks=false] expression

Print the Tile IR (MLIR) for all kernels compiled while evaluating the expression.
With `remarks=true`, also print `tileiras` optimization remarks for each kernel.

# Example
```julia
@device_code_tiled @cuda backend=cuTile blocks=grid vadd(a, b, c)
```
"""
macro device_code_tiled(ex...)
    hook = tile_hook((io, job; kwargs...) -> code_tiled(io, job; kwargs...))
    GPUCompiler.emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_structured [io=stdout] expression

Print the StructuredIRCode for all kernels compiled while evaluating the expression.

# Example
```julia
@device_code_structured @cuda backend=cuTile blocks=grid vadd(a, b, c)
```
"""
macro device_code_structured(ex...)
    hook = tile_hook((io, job; kwargs...) -> println(io, first(only(code_structured(job; kwargs...)))))
    GPUCompiler.emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_typed [io=stdout] expression

Print the typed Julia IR for all kernels compiled while evaluating the expression.

# Example
```julia
@device_code_typed @cuda backend=cuTile blocks=grid vadd(a, b, c)
```
"""
macro device_code_typed(ex...)
    hook = tile_hook((io, job; kwargs...) -> println(io, first(only(code_typed(job)))))
    GPUCompiler.emit_hooked_compilation(hook, ex...)
end
