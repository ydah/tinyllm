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
  ENV.fetch("TINY_LLM_RUNTIME_LR", FULL_PROFILE ? LEARNING_RATE.to_s : "0.01")
)
RUNTIME_EPOCHS = Integer(
  ENV.fetch("TINY_LLM_RUNTIME_EPOCHS", FULL_PROFILE ? EPOCHS.to_s : "120")
)
RUNTIME_BATCH_SIZE = Integer(
  ENV.fetch("TINY_LLM_RUNTIME_BATCH_SIZE", FULL_PROFILE ? BATCH_SIZE.to_s : "4")
)

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

def assert_close!(actual, expected, tolerance, label)
  difference = (actual - expected).abs
  return if difference <= tolerance

  raise "#{label} expected #{expected}, got #{actual} (diff=#{difference})"
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

def print_runtime_profile
  puts "=== Runtime Profile ==="
  puts "profile: #{FULL_PROFILE ? 'full' : 'demo'}"
  puts "embed_dim=#{RUNTIME_EMBED_DIM}, context_len=#{RUNTIME_CONTEXT_LEN}, hidden_dim=#{RUNTIME_HIDDEN_DIM}"
  puts "epochs=#{RUNTIME_EPOCHS}, batch_size=#{RUNTIME_BATCH_SIZE}, lr=#{RUNTIME_LEARNING_RATE}"
  puts
end

if __FILE__ == $PROGRAM_NAME
  print_runtime_profile
  run_tensor_sanity_check!
  run_tokenizer_sanity_check!
  puts "Foundation checks passed."
end
