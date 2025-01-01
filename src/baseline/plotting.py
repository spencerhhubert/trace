def countWeightsAffectingBaselineOutput(model):
    total_params = 0
    param_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            param_counts[name] = count
            total_params += count

    print("baseline:")
    print(f"{total_params} total parameters")
    for name, count in param_counts.items():
        print(f"{name}: {count}")

    return total_params
