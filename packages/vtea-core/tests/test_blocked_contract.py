"""Every step declares what it does to memory, and the declarations have to
stay true as steps are added - a step with no contract plans as if it were
free, which is the one failure mode a table like this has."""

import pytest

from vtea_core.blocked import (
    ACCUMULATE,
    APPROXIMATE,
    BLOCK_MODES,
    DEFAULT_SCALING,
    ELEMENTWISE,
    EXACT,
    EXACT_WITH_HALO,
    GLOBAL_STAT,
    NEIGHBORHOOD,
    OBJECT_LOCAL,
    TABLE,
    HaloSpec,
    MemoryBudget,
    Scaling,
    plan_for_steps,
    step_costs,
)
from vtea_core.blocked.contract import DEFAULT_OBJECT_EXTENT_VOXELS
from vtea_core.data import Spacing
from vtea_core.data.spacing import UNKNOWN
from vtea_core.workflow import Step
from vtea_core.workflow.registry import STEP_REGISTRY
from vtea_core.workflow.wiring import scaling_for

GB = 1024**3

REGISTERED = [
    (category, name) for category, functions in STEP_REGISTRY.items() for name in functions
]


@pytest.mark.parametrize(("category", "function_name"), REGISTERED)
def test_every_registered_step_declares_its_scaling(category, function_name):
    scaling = scaling_for(category, function_name)
    assert scaling.mode in BLOCK_MODES
    # A voxel step that claims to cost nothing per voxel would divide by
    # zero in the planner, and a table step that claims a per-voxel cost
    # would shrink every tile for work that never touches an image.
    if scaling.is_voxel_scaled:
        assert scaling.bytes_per_voxel > 0
    else:
        assert scaling.bytes_per_voxel == 0


@pytest.mark.parametrize(("category", "function_name"), REGISTERED)
def test_a_halo_only_belongs_to_a_step_that_reaches(category, function_name):
    scaling = scaling_for(category, function_name)
    if scaling.mode in (ELEMENTWISE, TABLE, GLOBAL_STAT, ACCUMULATE):
        assert scaling.halo.is_none, f"{category}.{function_name} declares a halo it cannot need"


@pytest.mark.parametrize(("category", "function_name"), REGISTERED)
def test_a_neighborhood_step_declares_how_far_it_reaches(category, function_name):
    scaling = scaling_for(category, function_name)
    if scaling.mode is NEIGHBORHOOD:
        assert not scaling.halo.is_none


@pytest.mark.parametrize(("category", "function_name"), REGISTERED)
def test_a_halo_parameter_is_a_real_parameter(category, function_name):
    """A halo that follows `sigma` is worthless if the argument is called
    something else - it silently resolves to zero."""
    import inspect

    from vtea_core.workflow.registry import get_step_function

    scaling = scaling_for(category, function_name)
    for variant in (scaling, *scaling.variants.values()):
        if variant.halo.param is None:
            continue
        parameters = inspect.signature(get_step_function(category, function_name)).parameters
        assert variant.halo.param in parameters, (
            f"{category}.{function_name} takes its halo from '{variant.halo.param}', "
            f"which is not one of its arguments: {list(parameters)}"
        )


@pytest.mark.parametrize(("category", "function_name"), REGISTERED)
def test_a_variant_keys_on_a_real_parameter(category, function_name):
    import inspect

    from vtea_core.workflow.registry import get_step_function

    scaling = scaling_for(category, function_name)
    if scaling.variant_param is None:
        return
    parameters = inspect.signature(get_step_function(category, function_name)).parameters
    assert scaling.variant_param in parameters


@pytest.mark.parametrize(("category", "function_name"), REGISTERED)
def test_a_variants_base_matches_the_functions_own_default(category, function_name):
    """`Scaling.resolve` cannot see a parameter the caller left alone, so
    the base entry has to be the one the function would actually take."""
    import inspect

    from vtea_core.workflow.registry import get_step_function

    scaling = scaling_for(category, function_name)
    if not scaling.variants:
        return
    parameter = inspect.signature(get_step_function(category, function_name)).parameters[
        scaling.variant_param
    ]
    default = parameter.default
    if default is inspect.Parameter.empty or str(default) not in scaling.variants:
        return
    assert scaling.resolve({}).mode == scaling.variants[str(default)].mode


def test_an_uncharacterised_step_gets_a_neutral_contract():
    # A third-party step registered through the entry-point groups is one
    # nobody has measured. Planning around it conservatively beats refusing
    # to plan.
    assert scaling_for("nonexistent", "step") is DEFAULT_SCALING


def test_threshold_is_a_different_step_depending_on_its_method():
    scaling = scaling_for("segmentation", "threshold_mask")
    assert scaling.resolve({"method": "fixed"}).mode == ELEMENTWISE
    assert scaling.resolve({"method": "otsu"}).mode == GLOBAL_STAT
    assert scaling.resolve({"method": "percentile"}).mode == GLOBAL_STAT
    # Left alone, it is whatever the function's own default does.
    assert scaling.resolve({}).mode == ELEMENTWISE


def test_clahe_is_admitted_to_be_approximate():
    scaling = scaling_for("imageprocessing", "enhance_contrast")
    assert scaling.resolve({"method": "normalize"}).exactness == EXACT
    assert scaling.resolve({"method": "equalize"}).exactness == APPROXIMATE


def test_cellpose_takes_its_halo_from_the_diameter_the_user_already_set():
    scaling = scaling_for("segmentation", "cellpose_segmentation")
    assert scaling.halo.resolve({"diameter": 30.0}, ndim=3) == (45, 45, 45)
    # And has a floor for when the diameter is left to the model to guess.
    assert min(scaling.halo.resolve({"diameter": None}, ndim=3)) >= 32


def test_a_physical_halo_is_anisotropic_when_the_voxels_are():
    # A 5 um dilation is 2.5 voxels along a 2 um z-step and 25 in x. A
    # scalar halo would be wrong on both axes at once.
    spacing = Spacing((2.0, 0.2, 0.2))
    halo = scaling_for("segmentation", "expand_labels").halo
    assert halo.resolve({"distance": 5.0}, spacing=spacing, ndim=3) == (3, 25, 25)


def test_a_physical_halo_falls_back_to_voxels_when_nobody_measured():
    unknown = Spacing((1.0, 1.0, 1.0), source=UNKNOWN)
    halo = scaling_for("segmentation", "expand_labels").halo
    assert halo.resolve({"distance": 5.0}, spacing=unknown, ndim=3) == (5, 5, 5)
    assert halo.resolve({"distance": 5.0}, spacing=None, ndim=3) == (5, 5, 5)


def test_object_extent_and_a_physical_parameter_do_not_contaminate_each_other():
    # label_shell reaches out by a physical parameter and in by however
    # deep the object is. Adding the two before converting would divide a
    # voxel count by the voxel size.
    spacing = Spacing((2.0, 0.2, 0.2))
    halo = scaling_for("segmentation", "label_shell").halo
    assert halo.resolve({"outward": 1.0}, spacing=spacing, ndim=3, object_extent=10) == (10, 10, 10)
    # ...and the physical term wins where it is the larger of the two.
    assert halo.resolve({"outward": 4.0}, spacing=spacing, ndim=3, object_extent=10) == (10, 20, 20)


def test_a_per_axis_parameter_gives_a_per_axis_halo():
    halo = scaling_for("imageprocessing", "gaussian_blur").halo
    assert halo.resolve({"sigma": (1.0, 2.0, 3.0)}, ndim=3) == (4, 8, 12)
    assert halo.resolve({"sigma": 2.0}, ndim=3) == (8, 8, 8)


def test_a_per_axis_parameter_of_the_wrong_length_is_an_error():
    halo = scaling_for("imageprocessing", "gaussian_blur").halo
    with pytest.raises(ValueError, match="does not fit"):
        halo.resolve({"sigma": (1.0, 2.0)}, ndim=3)


def test_an_object_sized_halo_falls_back_to_a_stated_default():
    halo = HaloSpec(object_extent=True)
    assert halo.resolve({}, ndim=3) == (DEFAULT_OBJECT_EXTENT_VOXELS,) * 3
    assert halo.resolve({}, ndim=3, object_extent=20) == (20, 20, 20)


def test_a_scaling_validates_its_own_vocabulary():
    with pytest.raises(ValueError, match="block mode"):
        Scaling(mode="sideways")
    with pytest.raises(ValueError, match="exactness"):
        Scaling(exactness="probably")
    with pytest.raises(ValueError, match="variant_param"):
        Scaling(variants={"a": Scaling()})


# -- how a pipeline's steps combine into one plan ------------------------


def a_step(category, function_name, **params):
    return Step.for_function(category, function_name, params=params)


def test_the_heaviest_step_sets_the_tile_size():
    steps = [
        a_step("imageprocessing", "gaussian_blur", sigma=1.0),
        a_step("segmentation", "watershed_split"),
        a_step("clustering", "kmeans", n_clusters=5),
    ]
    plan = plan_for_steps(steps, (512, 512, 512), budget=MemoryBudget(GB))
    assert plan.bound_by.startswith("watershed_split")
    assert plan.bytes_per_voxel == scaling_for("segmentation", "watershed_split").bytes_per_voxel


def test_the_halo_is_the_widest_any_step_needs():
    steps = [
        a_step("imageprocessing", "gaussian_blur", sigma=1.0),  # halo 4
        a_step("segmentation", "expand_labels", distance=20.0),  # halo 20
    ]
    plan = plan_for_steps(steps, (256, 256, 256), budget=MemoryBudget(8 * GB))
    assert plan.halo == (20, 20, 20)


def test_a_table_only_pipeline_does_not_shrink_the_tiles():
    # Clustering a measurement table is real work, but it is not a reason
    # to divide up an image nobody is reading.
    steps = [a_step("clustering", "kmeans", n_clusters=5), a_step("reduction", "pca")]
    plan = plan_for_steps(steps, (256, 256, 256), budget=MemoryBudget(GB))
    assert plan.is_single_tile
    assert any("no step" in note for note in plan.warnings())


def test_an_assumed_halo_is_flagged_and_a_stated_one_is_not():
    steps = [a_step("segmentation", "watershed_split")]
    shape = (256, 256, 256)
    assumed = plan_for_steps(steps, shape, budget=MemoryBudget(8 * GB))
    assert assumed.requires_verification
    stated = plan_for_steps(steps, shape, budget=MemoryBudget(8 * GB), object_extent=20)
    assert not stated.requires_verification
    assert stated.halo == (20, 20, 20)


def test_approximate_steps_are_named_on_the_plan():
    steps = [a_step("imageprocessing", "subtract_background", radius=10)]
    plan = plan_for_steps(steps, (256, 256, 256), budget=MemoryBudget(8 * GB))
    assert plan.approximate_steps == (steps[0].name,)
    assert any("same answer tiled as whole" in note for note in plan.warnings())


def test_a_channel_axis_is_held_out_of_the_tiling():
    steps = [a_step("measurements", "extract_measurements_by_channel")]
    plan = plan_for_steps(
        steps, (4, 256, 256, 256), budget=MemoryBudget(GB), tiled_axes=(1, 2, 3)
    )
    assert plan.tile[0] == 4
    assert plan.halo[0] == 0


def test_a_spatial_halo_lands_on_the_spatial_axes_of_a_4d_array():
    steps = [a_step("imageprocessing", "gaussian_blur", sigma=2.0)]
    plan = plan_for_steps(
        steps, (4, 256, 256, 256), budget=MemoryBudget(GB), tiled_axes=(1, 2, 3)
    )
    assert plan.halo == (0, 8, 8, 8)


def test_step_costs_explains_itself_per_step():
    steps = [
        a_step("imageprocessing", "gaussian_blur", sigma=2.0),
        a_step("clustering", "kmeans", n_clusters=3),
    ]
    costs = {cost.name: cost for cost in step_costs(steps)}
    blur = costs[steps[0].name]
    assert blur.halo == (8, 8, 8)
    assert blur.is_voxel_scaled
    assert not costs[steps[1].name].is_voxel_scaled


def test_an_empty_pipeline_plans_one_tile():
    plan = plan_for_steps([], (64, 64, 64), budget=MemoryBudget(GB))
    assert plan.is_single_tile


@pytest.mark.parametrize(("category", "function_name"), REGISTERED)
def test_every_step_can_be_planned_with_its_defaults(category, function_name):
    """The end-to-end guard: adding a step to the registry must not be able
    to break planning, whatever it declares."""
    step = Step.for_function(category, function_name)
    plan = plan_for_steps([step], (128, 128, 128), budget=MemoryBudget(8 * GB))
    assert plan.n_tiles >= 1
    assert plan.describe()


def test_exactness_vocabulary_is_used_as_documented():
    modes = {scaling_for(c, f).exactness for c, f in REGISTERED}
    assert modes <= {EXACT, EXACT_WITH_HALO, APPROXIMATE}
    # And the interesting ones are actually present, so a future refactor
    # that quietly flattens everything to EXACT gets caught.
    assert EXACT_WITH_HALO in modes
    assert APPROXIMATE in modes


def test_object_local_steps_exist_and_declare_a_reach():
    local = [(c, f) for c, f in REGISTERED if scaling_for(c, f).mode == OBJECT_LOCAL]
    assert local
    for category, function_name in local:
        assert not scaling_for(category, function_name).halo.is_none
