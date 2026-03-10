# frozen_string_literal: true

# -----------------------------------------------------------------------------
# Hyperparameters from the project specification.
# These are the "full" values described in .idea/spec.md.
# -----------------------------------------------------------------------------
EMBED_DIM = 32
CONTEXT_LEN = 32
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 500
BATCH_SIZE = 16

# Numerical safety values used throughout the implementation.
EPSILON = 1e-5
LOG_EPSILON = 1e-10

# -----------------------------------------------------------------------------
# Runtime profile
#
# A scalar autograd engine in pure Ruby becomes very slow with the full profile.
# The code keeps the spec constants above, but defaults to a smaller demo profile
# so `ruby tiny_llm.rb` remains practical as a learning exercise.
# Set TINY_LLM_PROFILE=full to run with the full spec values instead.
# -----------------------------------------------------------------------------
FULL_PROFILE = ENV["TINY_LLM_PROFILE"] == "full"

RUNTIME_EMBED_DIM = Integer(
  ENV.fetch("TINY_LLM_RUNTIME_EMBED_DIM", FULL_PROFILE ? EMBED_DIM.to_s : "8")
)
RUNTIME_CONTEXT_LEN = Integer(
  ENV.fetch("TINY_LLM_RUNTIME_CONTEXT_LEN", FULL_PROFILE ? CONTEXT_LEN.to_s : "8")
)
RUNTIME_HIDDEN_DIM = Integer(
  ENV.fetch("TINY_LLM_RUNTIME_HIDDEN_DIM", FULL_PROFILE ? HIDDEN_DIM.to_s : "16")
)
RUNTIME_LEARNING_RATE = Float(
  ENV.fetch("TINY_LLM_RUNTIME_LR", FULL_PROFILE ? LEARNING_RATE.to_s : "0.03")
)
RUNTIME_EPOCHS = Integer(
  ENV.fetch("TINY_LLM_RUNTIME_EPOCHS", FULL_PROFILE ? EPOCHS.to_s : "180")
)
RUNTIME_BATCH_SIZE = Integer(
  ENV.fetch("TINY_LLM_RUNTIME_BATCH_SIZE", FULL_PROFILE ? BATCH_SIZE.to_s : "4")
)
RUNTIME_SEED = Integer(ENV.fetch("TINY_LLM_SEED", "3"))

srand(RUNTIME_SEED)

class Tensor
  attr_accessor :value, :grad
  attr_reader :children, :op

  def self.scalar(value)
    new(value.to_f)
  end

  def self.wrap(value)
    value.is_a?(Tensor) ? value : scalar(value)
  end

  def initialize(value, children = [], op = "", &backward_fn)
    @value = value.to_f
    @grad = 0.0
    @children = children
    @op = op
    @backward_fn = backward_fn || proc {}
  end

  def +(other)
    other = Tensor.wrap(other)
    out = Tensor.new(@value + other.value, [self, other], "+")
    out.set_backward_fn do
      @grad += out.grad
      other.grad += out.grad
    end
    out
  end

  def *(other)
    other = Tensor.wrap(other)
    out = Tensor.new(@value * other.value, [self, other], "*")
    out.set_backward_fn do
      @grad += other.value * out.grad
      other.grad += @value * out.grad
    end
    out
  end

  def -(other)
    self + (Tensor.wrap(other) * -1.0)
  end

  def /(other)
    self * (Tensor.wrap(other)**-1.0)
  end

  def **(power)
    power = power.to_f
    out = Tensor.new(@value**power, [self], "**#{power}")
    out.set_backward_fn do
      @grad += power * (@value**(power - 1.0)) * out.grad
    end
    out
  end

  def exp
    out = Tensor.new(Math.exp(@value), [self], "exp")
    out.set_backward_fn do
      @grad += out.value * out.grad
    end
    out
  end

  def log
    clamped = [@value, LOG_EPSILON].max
    out = Tensor.new(Math.log(clamped), [self], "log")
    out.set_backward_fn do
      derivative = @value > LOG_EPSILON ? 1.0 / clamped : 0.0
      @grad += derivative * out.grad
    end
    out
  end

  def relu
    out = Tensor.new(@value.positive? ? @value : 0.0, [self], "relu")
    out.set_backward_fn do
      @grad += (@value.positive? ? 1.0 : 0.0) * out.grad
    end
    out
  end

  def tanh
    tanh_value = Math.tanh(@value)
    out = Tensor.new(tanh_value, [self], "tanh")
    out.set_backward_fn do
      @grad += (1.0 - tanh_value**2) * out.grad
    end
    out
  end

  def -@
    self * -1.0
  end

  def coerce(other)
    [Tensor.wrap(other), self]
  end

  def zero_grad!
    @grad = 0.0
  end

  def backward
    ordered = []
    visited = {}
    build_topology(self, visited, ordered)
    @grad = 1.0
    ordered.reverse_each(&:run_backward!)
  end

  def inspect
    format("Tensor(value=%0.4f, grad=%0.4f, op=%s)", @value, @grad, @op)
  end

  def set_backward_fn(&block)
    @backward_fn = block
  end

  def run_backward!
    @backward_fn.call
  end

  private

  def build_topology(node, visited, ordered)
    return if visited[node.object_id]

    visited[node.object_id] = true
    node.children.each do |child|
      build_topology(child, visited, ordered)
    end
    ordered << node
  end
end

class Tokenizer
  def initialize(text)
    chars = text.each_char.to_a.uniq.sort
    @char_to_id = chars.each_with_index.to_h
    @id_to_char = @char_to_id.invert
    @vocab_size = chars.length
  end

  def encode(text)
    text.each_char.map do |char|
      @char_to_id.fetch(char) do
        raise ArgumentError, "Unknown character for tokenizer: #{char.inspect}"
      end
    end
  end

  def decode(ids)
    ids.map do |id|
      @id_to_char.fetch(id) do
        raise ArgumentError, "Unknown token id for tokenizer: #{id.inspect}"
      end
    end.join
  end

  attr_reader :vocab_size
end

def xavier_value(fan_in)
  scale = 1.0 / Math.sqrt(fan_in.to_f)
  (rand * 2.0 * scale) - scale
end

def tensor_vector(size, fill = 0.0)
  Array.new(size) { Tensor.new(fill) }
end

def tensor_matrix(rows, cols, fan_in = cols)
  Array.new(rows) { Array.new(cols) { Tensor.new(xavier_value(fan_in)) } }
end

def transpose(matrix)
  return [] if matrix.empty?

  Array.new(matrix.first.length) do |column_index|
    Array.new(matrix.length) { |row_index| matrix[row_index][column_index] }
  end
end

def matmul(a, b)
  raise ArgumentError, "Cannot multiply empty matrices" if a.empty? || b.empty?

  shared_dim = a.first.length
  raise ArgumentError, "Matrix dimension mismatch" unless shared_dim == b.length

  result = Array.new(a.length) { Array.new(b.first.length) }

  a.length.times do |row_index|
    b.first.length.times do |column_index|
      sum = Tensor.scalar(0.0)
      shared_dim.times do |shared_index|
        sum += a[row_index][shared_index] * b[shared_index][column_index]
      end
      result[row_index][column_index] = sum
    end
  end

  result
end

def linear(row, weight, bias)
  raise ArgumentError, "Input length and weight rows must match" unless row.length == weight.length

  Array.new(weight.first.length) do |output_index|
    sum = bias[output_index]
    row.length.times do |input_index|
      sum += row[input_index] * weight[input_index][output_index]
    end
    sum
  end
end

def add_vectors(a, b)
  raise ArgumentError, "Vector sequence length mismatch" unless a.length == b.length
  return [] if a.empty?
  raise ArgumentError, "Vector width mismatch" unless a.first.length == b.first.length

  Array.new(a.length) do |row_index|
    Array.new(a.first.length) do |column_index|
      a[row_index][column_index] + b[row_index][column_index]
    end
  end
end

def softmax(logits)
  max_value = logits.map(&:value).max
  shifted = logits.map { |logit| logit - max_value }
  exp_values = shifted.map(&:exp)
  sum_exp = exp_values.reduce(Tensor.scalar(0.0), :+)
  exp_values.map { |value| value / sum_exp }
end

def average_tensor(values)
  raise ArgumentError, "Cannot average an empty array" if values.empty?

  values.reduce(Tensor.scalar(0.0), :+) / values.length.to_f
end

def shape_of(matrix)
  [matrix.length, matrix.empty? ? 0 : matrix.first.length]
end

class Embedding
  def initialize(vocab_size, embed_dim)
    @weight = tensor_matrix(vocab_size, embed_dim, embed_dim)
  end

  def forward(token_ids)
    token_ids.map { |token_id| @weight.fetch(token_id) }
  end

  def params
    @weight.flatten
  end
end

class PositionEmbedding
  def initialize(context_len, embed_dim)
    @weight = tensor_matrix(context_len, embed_dim, embed_dim)
  end

  def forward(seq_len)
    raise ArgumentError, "Sequence length #{seq_len} exceeds context #{@weight.length}" if seq_len > @weight.length

    @weight.first(seq_len)
  end

  def params
    @weight.flatten
  end
end

class LayerNorm
  def initialize(dim)
    @gamma = tensor_vector(dim, 1.0)
    @beta = tensor_vector(dim, 0.0)
  end

  def forward(x)
    x.map do |row|
      mean = average_tensor(row)
      variance = average_tensor(row.map { |value| (value - mean)**2.0 })
      denominator = (variance + EPSILON)**0.5

      row.each_with_index.map do |value, index|
        normalized = (value - mean) / denominator
        (@gamma[index] * normalized) + @beta[index]
      end
    end
  end

  def params
    @gamma + @beta
  end
end

class SelfAttention
  def initialize(embed_dim)
    @embed_dim = embed_dim
    @wq = tensor_matrix(embed_dim, embed_dim, embed_dim)
    @wk = tensor_matrix(embed_dim, embed_dim, embed_dim)
    @wv = tensor_matrix(embed_dim, embed_dim, embed_dim)
  end

  def forward(x)
    q = matmul(x, @wq)
    k = matmul(x, @wk)
    v = matmul(x, @wv)

    scores = matmul(q, transpose(k))
    scaled_scores = scores.map do |row|
      row.map { |score| score / Math.sqrt(@embed_dim.to_f) }
    end

    masked_scores = scaled_scores.each_with_index.map do |row, row_index|
      row.each_with_index.map do |score, column_index|
        column_index > row_index ? (score + -1e9) : score
      end
    end

    attention_weights = masked_scores.map { |row| softmax(row) }
    matmul(attention_weights, v)
  end

  def params
    @wq.flatten + @wk.flatten + @wv.flatten
  end
end

class FeedForward
  def initialize(embed_dim, hidden_dim)
    @w1 = tensor_matrix(embed_dim, hidden_dim, embed_dim)
    @b1 = tensor_vector(hidden_dim, 0.0)
    @w2 = tensor_matrix(hidden_dim, embed_dim, hidden_dim)
    @b2 = tensor_vector(embed_dim, 0.0)
  end

  def forward(x)
    x.map do |row|
      hidden = linear(row, @w1, @b1).map(&:relu)
      linear(hidden, @w2, @b2)
    end
  end

  def params
    @w1.flatten + @b1 + @w2.flatten + @b2
  end
end

class TransformerBlock
  def initialize(embed_dim, hidden_dim)
    @ln1 = LayerNorm.new(embed_dim)
    @attn = SelfAttention.new(embed_dim)
    @ln2 = LayerNorm.new(embed_dim)
    @ffn = FeedForward.new(embed_dim, hidden_dim)
  end

  def forward(x)
    normalized = @ln1.forward(x)
    attended = @attn.forward(normalized)
    residual = add_vectors(x, attended)

    normalized_residual = @ln2.forward(residual)
    feed_forward = @ffn.forward(normalized_residual)
    add_vectors(residual, feed_forward)
  end

  def params
    @ln1.params + @attn.params + @ln2.params + @ffn.params
  end
end

class TinyLLM
  attr_reader :context_len, :vocab_size

  def initialize(vocab_size, embed_dim, context_len, hidden_dim)
    @vocab_size = vocab_size
    @context_len = context_len
    @tok_emb = Embedding.new(vocab_size, embed_dim)
    @pos_emb = PositionEmbedding.new(context_len, embed_dim)
    @block = TransformerBlock.new(embed_dim, hidden_dim)
    @ln_final = LayerNorm.new(embed_dim)
    @output_w = tensor_matrix(embed_dim, vocab_size, embed_dim)
    @output_b = tensor_vector(vocab_size, 0.0)
  end

  def forward(token_ids)
    raise ArgumentError, "Input length exceeds context window" if token_ids.length > @context_len

    token_embeddings = @tok_emb.forward(token_ids)
    position_embeddings = @pos_emb.forward(token_ids.length)
    x = add_vectors(token_embeddings, position_embeddings)
    x = @block.forward(x)
    x = @ln_final.forward(x)
    x.map { |row| linear(row, @output_w, @output_b) }
  end

  def params
    @tok_emb.params + @pos_emb.params + @block.params + @ln_final.params + @output_w.flatten + @output_b
  end
end

def assert_close!(actual, expected, tolerance, label)
  difference = (actual - expected).abs
  return if difference <= tolerance

  raise "#{label} expected #{expected}, got #{actual} (diff=#{difference})"
end

def assert_shape!(matrix, expected_shape, label)
  actual_shape = shape_of(matrix)
  return if actual_shape == expected_shape

  raise "#{label} expected shape #{expected_shape.inspect}, got #{actual_shape.inspect}"
end

def run_tensor_sanity_check!
  a = Tensor.new(2.0)
  b = Tensor.new(3.0)
  c = (a * b) + b
  c.backward

  assert_close!(a.grad, 3.0, 1e-9, "Tensor grad for a")
  assert_close!(b.grad, 3.0, 1e-9, "Tensor grad for b")
end

def run_tokenizer_sanity_check!
  tokenizer = Tokenizer.new("hello world")
  ids = tokenizer.encode("hello")

  raise "Tokenizer decode mismatch" unless tokenizer.decode(ids) == "hello"
  raise "Tokenizer vocab size mismatch" unless tokenizer.vocab_size == 8
end

def run_shape_sanity_checks!(tokenizer)
  embedding = Embedding.new(tokenizer.vocab_size, RUNTIME_EMBED_DIM)
  position = PositionEmbedding.new(RUNTIME_CONTEXT_LEN, RUNTIME_EMBED_DIM)
  layer_norm = LayerNorm.new(RUNTIME_EMBED_DIM)
  attention = SelfAttention.new(RUNTIME_EMBED_DIM)
  feed_forward = FeedForward.new(RUNTIME_EMBED_DIM, RUNTIME_HIDDEN_DIM)
  block = TransformerBlock.new(RUNTIME_EMBED_DIM, RUNTIME_HIDDEN_DIM)
  model = TinyLLM.new(tokenizer.vocab_size, RUNTIME_EMBED_DIM, RUNTIME_CONTEXT_LEN, RUNTIME_HIDDEN_DIM)

  token_vectors = embedding.forward([0, 1, 2])
  assert_shape!(token_vectors, [3, RUNTIME_EMBED_DIM], "Embedding output")

  position_vectors = position.forward(3)
  assert_shape!(position_vectors, [3, RUNTIME_EMBED_DIM], "PositionEmbedding output")

  normalized = layer_norm.forward(add_vectors(token_vectors, position_vectors))
  assert_shape!(normalized, [3, RUNTIME_EMBED_DIM], "LayerNorm output")

  attended = attention.forward(normalized)
  assert_shape!(attended, [3, RUNTIME_EMBED_DIM], "SelfAttention output")

  forwarded = feed_forward.forward(attended)
  assert_shape!(forwarded, [3, RUNTIME_EMBED_DIM], "FeedForward output")

  blocked = block.forward(normalized)
  assert_shape!(blocked, [3, RUNTIME_EMBED_DIM], "TransformerBlock output")

  logits = model.forward([0, 1, 2])
  assert_shape!(logits, [3, tokenizer.vocab_size], "TinyLLM logits")
end

def build_training_examples(tokens, context_len)
  max_start = tokens.length - context_len - 1
  raise ArgumentError, "Training data is too short for context length #{context_len}" if max_start.negative?

  (0..max_start).map do |start_index|
    [
      tokens[start_index, context_len],
      tokens[start_index + 1, context_len]
    ]
  end
end

def cross_entropy_loss(logits, targets)
  losses = logits.each_with_index.map do |row, index|
    probabilities = softmax(row)
    -probabilities.fetch(targets.fetch(index)).log
  end
  average_tensor(losses)
end

def train(model, data, tokenizer, epochs, lr, batch_size)
  tokens = tokenizer.encode(data)
  examples = build_training_examples(tokens, model.context_len)
  all_params = model.params
  report_interval = FULL_PROFILE ? 50 : [epochs / 6, 1].max
  loss_history = []
  best_loss = Float::INFINITY
  best_values = all_params.map(&:value)

  epochs.times do |epoch|
    all_params.each(&:zero_grad!)
    current_values = all_params.map(&:value)

    batch = Array.new(batch_size) { examples.fetch(rand(examples.length)) }
    batch_loss = batch.reduce(Tensor.scalar(0.0)) do |loss_sum, (input_ids, target_ids)|
      loss_sum + cross_entropy_loss(model.forward(input_ids), target_ids)
    end / batch.length.to_f

    batch_loss.backward

    if batch_loss.value < best_loss
      best_loss = batch_loss.value
      best_values = current_values
    end

    all_params.each do |param|
      param.value -= lr * param.grad
      param.zero_grad!
    end

    loss_history << batch_loss.value
    epoch_number = epoch + 1
    should_report = epoch.zero? || epoch_number % report_interval == 0 || epoch_number == epochs
    puts format("epoch %3d/%3d  loss=%0.4f", epoch_number, epochs, batch_loss.value) if should_report
  end

  all_params.zip(best_values) do |param, best_value|
    param.value = best_value
  end

  {
    history: loss_history,
    best_loss: best_loss
  }
end

def sample_from_probs(probabilities)
  threshold = rand
  cumulative = 0.0

  probabilities.each_with_index do |probability, index|
    cumulative += probability
    return index if threshold < cumulative
  end

  probabilities.length - 1
end

def generate(model, tokenizer, seed_text, length, temperature = 1.0)
  temperature = [temperature, 0.1].max
  tokens = tokenizer.encode(seed_text)

  length.times do
    context_tokens = tokens.last(model.context_len)
    logits = model.forward(context_tokens)
    scaled_logits = logits.last.map { |logit| logit / temperature }
    probabilities = softmax(scaled_logits).map(&:value)
    tokens << sample_from_probs(probabilities)
  end

  tokenizer.decode(tokens)
end

def print_runtime_profile
  puts "=== Runtime Profile ==="
  puts "profile: #{FULL_PROFILE ? 'full' : 'demo'}"
  puts "embed_dim=#{RUNTIME_EMBED_DIM}, context_len=#{RUNTIME_CONTEXT_LEN}, hidden_dim=#{RUNTIME_HIDDEN_DIM}"
  puts "epochs=#{RUNTIME_EPOCHS}, batch_size=#{RUNTIME_BATCH_SIZE}, lr=#{RUNTIME_LEARNING_RATE}"
  puts "seed=#{RUNTIME_SEED}"
  puts
end

TRAIN_DATA = <<~TEXT
  To be or not to be, that is the question.
  Whether tis nobler in the mind to suffer
  the slings and arrows of outrageous fortune,
  or to take arms against a sea of troubles.
  To be or not to be, that is the question.
  Whether tis nobler in the mind to suffer
  the slings and arrows of outrageous fortune,
  or to take arms against a sea of troubles.
TEXT

if __FILE__ == $PROGRAM_NAME
  print_runtime_profile
  run_tensor_sanity_check!
  run_tokenizer_sanity_check!
  tokenizer = Tokenizer.new(TRAIN_DATA)
  run_shape_sanity_checks!(tokenizer)

  model = TinyLLM.new(
    tokenizer.vocab_size,
    RUNTIME_EMBED_DIM,
    [RUNTIME_CONTEXT_LEN, tokenizer.encode(TRAIN_DATA).length - 1].min,
    RUNTIME_HIDDEN_DIM
  )

  puts "Foundation checks passed."
  puts "Parameter count: #{model.params.length}"
  puts
  puts "=== Training ==="
  training_result = train(
    model,
    TRAIN_DATA,
    tokenizer,
    RUNTIME_EPOCHS,
    RUNTIME_LEARNING_RATE,
    RUNTIME_BATCH_SIZE
  )
  puts "Final loss: #{format('%0.4f', training_result[:history].last)}"
  puts "Best loss:  #{format('%0.4f', training_result[:best_loss])}"
  puts
  puts "=== Generation ==="
  puts generate(model, tokenizer, "To be", 100, 0.8)
end
