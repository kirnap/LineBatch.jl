module LineBatch

export StreamData

import Base: start, next, done

type StreamData
    
    vocabulary::Dict
    linewordcount::Dict
    seqdict::Dict
    seqcount::Dict
    batchsize::Int
    issparsed::Bool
end


""" Construct the meta knowledge about the directory struct """
function StreamData(inputfile::AbstractString; vocabulary=nothing, batchsize=nothing, issparsed=true)
    (batchsize == nothing) && error("Please specify batchsize")
    data_vocab = (vocabulary == nothing ? createvocab(inputfile) : vocabulary)
    linewordcount = countwordlines(inputfile)
    cleandata!(linewordcount, batchsize)
    (isempty(linewordcount)) && error("There is no enough sentence with that batchsize=$batchsize")
    seqdict = linetosequence(inputfile, data_vocab)
    seqcount = get_sequencecount(linewordcount)
    StreamData(data_vocab, linewordcount, seqdict, seqcount, batchsize, issparsed)
end


function start(s::StreamData)
    linewordcount = copy(s.linewordcount)
    seqcount = copy(s.seqcount)
    state = (seqcount, linewordcount) 
    return state
end


function next(s::StreamData, state)
    (seqcount, linewordcount) = state
    seqlen = rand(collect(keys(seqcount))) 
    lines = getlines(linewordcount, seqlen)
    lines = lines[1:s.batchsize]
    _rmlines!(linewordcount, lines)
    _rmseqbatch!(seqcount, seqlen, s.batchsize)
    item = sentenbatch(s, lines, seqlen)
    return (item, state)
end


function done(s::StreamData, state)
    (seqcount, linewordcount) = state
    return isempty(seqcount)
end


""" To remove the lines that are already being used in the batches """
function _rmlines!(linewordcount::Dict, lines::Array)
    for line in lines
	delete!(linewordcount, line)
    end
end


""" substracts the minibatch size from the given seqcount[seqlen] item """
function _rmseqbatch!(seqcount::Dict, seqlen::Int, batchsize::Int)
    seqcount[seqlen] -= batchsize		
    (seqcount[seqlen] < batchsize) && (delete!(seqcount, seqlen))

end


""" Returns the dictionary that holds a key as a line number and value as a mapped integer sequence """
function linetosequence(inputfile::AbstractString, vocabulary::Dict)
    sequence = Dict()
    i=1
    open(inputfile) do file
	for line in eachline(file)
	    words = map(x->vocabulary[x], split(line))
	    sequence[i] = words
	    i += 1
	end
    end
    return sequence
end


"""Reads from the file and creates a vocabulary"""
function createvocab(inputfile::AbstractString)
    vocabulary = Dict{AbstractString, Int}("<s>"=> 1)
    open(inputfile) do file
        for line in eachline(file)
            words = split(line)
            for word in words; get!(vocabulary, word, 1+length(vocabulary));end
        end
    end
    if !("<unk>" in keys(vocabulary))
        vocabulary["<unk>"] = 1 + length(vocabulary)
    end
    vocabulary["</s>"] = 1 + length(vocabulary)
    return vocabulary
end


""" Returns a dictionary of the form linenumber => number of words in that line """
function countwordlines(inputfile::AbstractString)
    file = open(inputfile)
    linewordcount = Dict{Int, Int}(enumerate(map(x->length(split(x)), eachline(file))))
    return linewordcount
end


""" Returns an array of numbers that holds the linenumbers for a given number of words """
function getlines(linewordcount::Dict{Int,Int}, wordnumbers::Int)
    keyiterator = keys(filter((f,v) -> v==wordnumbers, linewordcount))
    return collect(keyiterator)
end


"""
	Creates minibatches of the sentences with the same length, the sparse representation also available. 
        e.g.,
        The dog ran
        The next person
        minibatch_1 = [the; the]
        minibatch_2 = [dog; next]
        minibatch_3 = [ran; person]
   	Be careful not the words are put but the word codes from the vocabulary are put

"""
function sentenbatch(s::StreamData, linenumbers::Array, seqlen::Int)
    data = Any[]
    vocablen = length(s.vocabulary)
    batchsize = s.batchsize
    sequencedict = s.seqdict
    for cursor=1:seqlen
	d = zeros(Float32, vocablen, batchsize)
	for i=1:batchsize
	    sentence = linenumbers[i]
	    index = sequencedict[sentence][cursor]
	    d[index, i] = 1
	end
	d_sparse = s.issparsed ? sparse(d) : d
	push!(data, d_sparse)
    end
    return data
end


""" Removes the  sentences that either do not have enough words or its amount is less than batchsize """
function cleandata!(linewordcount::Dict, batchsize::Int)
    for item in unique(values(linewordcount)) 
	lines  = getlines(linewordcount, item)
	if (length(lines) < batchsize || item == 1 || item == 2)
	    map(x->delete!(linewordcount, x), lines)
	end
    end
end


""" Returns a dictionary of seqlength => #oflines """
    function get_sequencecount(linewordcount::Dict)
	result = Dict{Int, Int}(map(x->(x,0), unique(values(linewordcount))))
	for value in values(linewordcount)
	    result[value] += 1
	end
	return result
    end

# TODO:
# if the word is out of vocab, <unk> it
# if mode is dev than do not get rid of the surplus sentences

end # module
