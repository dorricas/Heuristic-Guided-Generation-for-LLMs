from llama_3_model import Llama3Model 
import numpy as np

if __name__ == "__main__":
    llama3_ckpts = "/home/sumedh/meta-llama"
    llama_model = Llama3Model(llama3_ckpts, "1B", max_batch_size=3)

    main_str = "I am providing only one sentence."

    for i in range(20):
        output, probs = llama_model.generate([main_str], eos_token_id=['\n', 13], max_new_tokens=1, output_log_probs=True)
        print(main_str, output)
        probs  = llama_model.get_loglikelihood(main_str, [main_str + output[0]])
        main_str += output[0]
        print(main_str)
        print(output, np.exp(probs))
    
    #print(np.exp(llama_model.get_loglikelihood("The capital of UK is", ["The capital of UK is Paris", "The capital of UK is London", "The capital of UK is Moscow"])))


