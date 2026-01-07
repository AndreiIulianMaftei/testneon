// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/database/oldTimeCollection.hpp"

namespace NeoN::finiteVolume::cellCentred
{

OldTimeDocument::OldTimeDocument(const Document& doc) : doc_(doc) {}

OldTimeDocument::OldTimeDocument(
    std::string nextTime, std::string previousTime, std::string currentTime, int32_t level
)
    : doc_(Document(
        {{"nextTime", nextTime},
         {"previousTime", previousTime},
         {"currentTime", currentTime},
         {"level", level}}
    ))
{}

std::string& OldTimeDocument::nextTime() { return doc_.get<std::string>("nextTime"); }

const std::string& OldTimeDocument::nextTime() const { return doc_.get<std::string>("nextTime"); }

std::string& OldTimeDocument::previousTime() { return doc_.get<std::string>("previousTime"); }

const std::string& OldTimeDocument::previousTime() const
{
    return doc_.get<std::string>("previousTime");
}

std::string& OldTimeDocument::currentTime() { return doc_.get<std::string>("currentTime"); }

const std::string& OldTimeDocument::currentTime() const
{
    return doc_.get<std::string>("currentTime");
}

int32_t& OldTimeDocument::level() { return doc_.get<int32_t>("level"); }

const int32_t& OldTimeDocument::level() const { return doc_.get<int32_t>("level"); }

Document& OldTimeDocument::doc() { return doc_; }

const Document& OldTimeDocument::doc() const { return doc_; }


std::string OldTimeDocument::id() const { return doc_.id(); }

std::string OldTimeDocument::typeName() { return "OldTimeDocument"; }

OldTimeCollection::OldTimeCollection(
    Database& db, std::string name, std::string fieldCollectionName
)
    : CollectionMixin<OldTimeDocument>(db, name), fieldCollectionName_(fieldCollectionName)
{}

OldTimeDocument& OldTimeCollection::oldTimeDoc(const std::string& id) { return docs_.at(id); }

const OldTimeDocument& OldTimeCollection::oldTimeDoc(const std::string& id) const
{
    return docs_.at(id);
}

void OldTimeCollection::setCurrentVectorAndLevel(OldTimeDocument& oldTimeDoc)
{
    // find the document which has the previousTime identical to the nextTime
    // so the document on above in the chain
    std::string nextId = findPreviousTime(oldTimeDoc.nextTime());
    if (nextId == "") // not registered yet
    {
        oldTimeDoc.currentTime() = oldTimeDoc.nextTime();
        oldTimeDoc.level() = 1;
        return;
    }
    // get the next document and set the current field and level
    auto& nextDoc = docs_.at(nextId);
    oldTimeDoc.currentTime() = nextDoc.currentTime();
    oldTimeDoc.level() = nextDoc.level() + 1;
}

bool OldTimeCollection::contains(const std::string& id) const
{
    return docs_.contains(id);
    ;
}

bool OldTimeCollection::insert(const OldTimeDocument& otd)
{
    std::string id = otd.id();
    if (contains(id))
    {
        return false;
    }
    docs_.emplace(id, otd);
    return true;
}

std::string OldTimeCollection::findNextTime(std::string id) const
{
    auto keys = find([id](const Document& doc) { return doc.get<std::string>("nextTime") == id; });
    if (keys.size() == 1)
    {
        return keys[0];
    }
    return "";
}

std::string OldTimeCollection::findPreviousTime(std::string id) const
{
    auto keys =
        find([id](const Document& doc) { return doc.get<std::string>("previousTime") == id; });
    if (keys.size() == 1)
    {
        return keys[0];
    }
    return "";
}

OldTimeCollection&
OldTimeCollection::instance(Database& db, std::string name, std::string fieldCollectionName)
{
    Collection& col = db.insert(name, OldTimeCollection(db, name, fieldCollectionName));
    return col.as<OldTimeCollection>();
}

const OldTimeCollection& OldTimeCollection::instance(const Database& db, std::string name)
{
    const Collection& col = db.at(name);
    return col.as<OldTimeCollection>();
}

OldTimeCollection& OldTimeCollection::instance(VectorCollection& fieldCollection)
{
    std::string name = fieldCollection.name() + "_oldTime";
    return instance(fieldCollection.db(), name, fieldCollection.name());
}

const OldTimeCollection& OldTimeCollection::instance(const VectorCollection& fieldCollection)
{
    std::string name = fieldCollection.name() + "_oldTime";
    return instance(fieldCollection.db(), name);
}

template<typename VectorType>
void rotate(VectorType& field)
{
    /**
     * Rotate time levels for a field (OpenFOAM-style).
     *
     * Meaning of `level` (computed via oldTimeLevel(field)):
     *   level == 0 : no history exists yet
     *   level == 1 : φ^{n}   exists (one oldTime buffer)
     *   level >= 2 : φ^{n} and φ^{n-1} exist (old and oldOld)
     *
     * Startup sequence when rotate() is called at the BEGINNING of each time step:
     *
     *   Initial state (t = t0, before first rotate):
     *     field          = φ^{0}
     *     no oldTime buffers exist
     *
     *   First rotate() call (entering t1):
     *     - level == 0
     *     - φ^{n} buffer is allocated via oldTime(field)
     *     - data rotated: φ^{n} ← φ^{0}
     *     → history depth = 1  (BDF1 startup)
     *
     *   Second rotate() call (entering t2):
     *     - level == 1
     *     - φ^{n-1} buffer is allocated via oldTime(oldTime(field))
     *     - data rotated: φ^{n-1} ← φ^{n} = φ^{0}, φ^{n} ← φ^{1}
     *     → history depth = 2  (BDF2 becomes active)
     *
     *   Subsequent rotate() calls:
     *     - level >= 2
     *     - buffers reused (no further allocation)
     *     - data rotated each step: φ^{n-1} ← φ^{n}, φ^{n} ← φ^{current}
     *
     * Only two historical states are kept (φ^{n}, φ^{n-1}).
     */
    VectorCollection& fieldCollection = VectorCollection::instance(field);
    OldTimeCollection& oldTimeCollection = OldTimeCollection::instance(fieldCollection);

    const int level = oldTimeLevel(field);

    // Get or create phi^n (oldTime)
    VectorType& oldVector = oldTime(field);

    if (level == 1)
    {
        VectorType& oldOldVector = oldTime(oldVector);
    }
    if (level >= 2)
    {
        VectorType& oldOldVector = oldTime(oldVector);
        oldOldVector.internalVector() = oldVector.internalVector();
        oldOldVector.boundaryData() = oldVector.boundaryData();
    }

    // Rotate current -> old
    oldVector.internalVector() = field.internalVector();
    oldVector.boundaryData() = field.boundaryData();
}

template void NeoN::finiteVolume::cellCentred::rotate<NeoN::finiteVolume::cellCentred::VolumeField<
    NeoN::scalar>>(NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>&);

template void NeoN::finiteVolume::cellCentred::rotate<NeoN::finiteVolume::cellCentred::VolumeField<
    NeoN::Vec3>>(NeoN::finiteVolume::cellCentred::VolumeField<NeoN::Vec3>&);

template void NeoN::finiteVolume::cellCentred::rotate<NeoN::finiteVolume::cellCentred::SurfaceField<
    NeoN::scalar>>(NeoN::finiteVolume::cellCentred::SurfaceField<NeoN::scalar>&);

} // namespace NeoN::finiteVolume::cellCentred
