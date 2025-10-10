# Section 2.7 — Batching and Tensorization

## 🔑 Key Ideas
1. **Why batching?**  
   - Training with one sequence at a time is slow and noisy.  
   - Batching lets the model process many sequences in parallel, making training faster and more stable.

2. **From lists to tensors**  
   - Previously, `AIDatasetV1` gave us Python lists or individual tensors.  
   - Now, a `DataLoader` collects multiple samples and stacks them into a batch.  
   - Result shapes:  
     - `X`: `[batch_size, seq_len]`  
     - `Y`: `[batch_size, seq_len]`

3. **Example**  
   With `batch_size=2` and `seq_len=4`:  

   - Dataset might yield:  
     - `X₁ = [50256, 123, 45, 678]`  
     - `Y₁ = [123, 45, 678, 901]`  
     - `X₂ = [33, 44, 55, 66]`  
     - `Y₂ = [44, 55, 66, 77]`  

   - DataLoader batches them into tensors:  
     ```
     X = [[50256, 123, 45, 678],
          [   33,  44, 55,  66]]

     Y = [[123, 45, 678, 901],
          [ 44, 55,  66,  77]]
     ```

   - Both are now **rank-2 tensors** that can be fed to the model.

4. **Shift check**  
   - In every row, `Y` is just `X` shifted left by one token.  
   - That’s the supervised target for next-token prediction.

5. **Train vs. validation loaders**  
   - **Train loader:** shuffled batches for gradient updates.  
   - **Validation loader:** deterministic batches for evaluation.

---